"""Prospectors & Pirates bot (v2).

An object-oriented implementation of the heuristic described in
``docs_game/bot_algorithm_heuristic.txt``. The logic is split into small,
reusable pieces so a future strategy (e.g. a learned policy) can lean on the
same primitives:

* :class:`ShipUtils`       - stateless geometry / parsing helpers.
* :class:`CombatEvaluator` - attack / defence power estimation (skills-aware).
* :class:`GameContext`     - parses one ActionRequest into structured state.
* :class:`HeuristicStrategy` - the priority-ordered decision + action mask.

Only :func:`get_action` / :func:`get_heuristic_action` are part of the public
lambda contract; everything else is reusable building blocks.
"""

import logging
import logging.handlers
import math
import os
import queue
import sys
import time

def _to_response(payload):
    """Lightweight lambda response adapter.

    The lambda runtime accepts plain dicts. This adapter normalises common
    payload shapes and guarantees a valid action object is returned.
    """
    if isinstance(payload, dict):
        action = payload
    elif hasattr(payload, "to_dict") and callable(payload.to_dict):
        action = payload.to_dict()
    elif hasattr(payload, "__dict__"):
        action = {
            key: value
            for key, value in vars(payload).items()
            if not key.startswith("_")
        }
    else:
        action = None

    if not isinstance(action, dict):
        return {"actionType": "WAIT"}

    action_type = action.get("actionType")
    if not isinstance(action_type, str) or not action_type:
        return {"actionType": "WAIT"}

    return action

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Performance: never propagate to the root logger. Propagation would emit every
# per-tick trace to the console (stderr) synchronously on the decision hot
# path; the bot's logs are only ever consumed from the per-game files below.
logger.propagate = False


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _logging_enabled():
    """Whether per-game diagnostic logging is active.

    Logging is automatically DISABLED when running inside AWS Lambda -- the
    runtime sets ``AWS_LAMBDA_FUNCTION_NAME`` -- so the official/competition
    deployment pays zero logging overhead on the decision hot path. Overrides:

    * ``PNP_DISABLE_LOGGING`` truthy -> always off (e.g. to benchmark locally).
    * ``PNP_FORCE_LOGGING``  truthy -> always on, even on Lambda (to debug a
      live deployment); takes precedence over the auto-disable.
    """
    if _truthy(os.environ.get("PNP_DISABLE_LOGGING")):
        return False
    if _truthy(os.environ.get("PNP_FORCE_LOGGING")):
        return True
    return "AWS_LAMBDA_FUNCTION_NAME" not in os.environ


def _refresh_logging_enabled():
    """Recompute the logging flag from the environment and apply it.

    When disabled, ``logger.disabled`` short-circuits every inline ``logger``
    call cheaply, and the heavy log-builder helpers below early-return, so no
    log records, file handlers or writer threads are ever created.
    """
    global _LOGGING_ENABLED
    _LOGGING_ENABLED = _logging_enabled()
    logger.disabled = not _LOGGING_ENABLED
    return _LOGGING_ENABLED


_LOGGING_ENABLED = False
_refresh_logging_enabled()



# -----------------------------------
# Per-game file logging
# -----------------------------------
# Each game+round gets its own log file (named "<gameId>_<round>_game.log") so a
# single round's tick-by-tick trace can be inspected in isolation. The file
# handler is created once per game+round and reused across ticks; switching to a
# new game or round detaches the previous handler so logs never cross-
# contaminate. Everything here is best-effort -- any filesystem error is
# swallowed so logging never breaks the bot's decision.
#
# Performance: the file I/O is asynchronous. A QueueHandler (attached to the
# logger once) only enqueues records -- a cheap, non-blocking operation on the
# decision hot path -- while a background QueueListener thread drains the queue
# and performs the actual disk writes off the critical path.
_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_game_log_handlers = {}                       # (game_id, round) -> FileHandler
_active_game_log = {"key": None, "handler": None}
_endgame_logged = set()                       # (game_id, round) stats written
_round_start_logged = set()                   # (game_id, round) start state written

_log_queue = queue.SimpleQueue()              # records handed off to the writer thread
_queue_handler = logging.handlers.QueueHandler(_log_queue)
_queue_handler.setLevel(logging.INFO)
logger.addHandler(_queue_handler)
_queue_listener = None                        # background thread feeding the active file


def _log_dir():
    """Directory for per-game log files.

    Defaults to a ``logs`` folder next to this module (created on demand). Set
    PNP_LOG_DIR to override -- e.g. to ``/tmp`` on AWS Lambda, where only the
    temp dir is writable.
    """
    return os.environ.get("PNP_LOG_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs"
    )


def _ensure_game_log_file(game_id, round_no):
    """Attach a file handler writing this round's logs to ``<game_id>_<round>_game.log``."""
    global _queue_listener
    if not _LOGGING_ENABLED or game_id is None:
        return
    key = (game_id, round_no)
    if _active_game_log["key"] == key:
        return
    handler = _game_log_handlers.get(key)
    if handler is None:
        try:
            log_dir = _log_dir()
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"{game_id}_{round_no}_game.log")
            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            _game_log_handlers[key] = handler
        except OSError:
            logger.warning("Could not open log file for game %s round %s", game_id, round_no)
            _active_game_log["key"] = key
            _active_game_log["handler"] = None
            return
    # Repoint the background writer thread at this round's file. Stopping the
    # previous listener first flushes any records still queued for the prior
    # round into that round's own file, so logs never cross-contaminate.
    if _queue_listener is not None:
        _queue_listener.stop()
    _queue_listener = logging.handlers.QueueListener(
        _log_queue, handler, respect_handler_level=True
    )
    _queue_listener.start()
    _active_game_log["key"] = key
    _active_game_log["handler"] = handler


def _flush_game_logging():
    """Block until every queued record has been written to disk.

    The file writes are asynchronous, so callers that need to read a log file
    immediately (chiefly the tests) must drain the queue first. Stopping the
    listener joins the writer thread after the queue empties; we then restart a
    fresh listener so subsequent logging in the same round still flows.
    """
    global _queue_listener
    if _queue_listener is None:
        return
    _queue_listener.stop()
    handler = _active_game_log["handler"]
    if handler is None:
        _queue_listener = None
        return
    _queue_listener = logging.handlers.QueueListener(
        _log_queue, handler, respect_handler_level=True
    )
    _queue_listener.start()


def _reset_game_logging():
    """Stop the writer thread, close and forget all per-game file handlers (used by tests)."""
    global _queue_listener
    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None
    for handler in _game_log_handlers.values():
        try:
            handler.close()
        except Exception:  # pragma: no cover - close is best-effort
            pass
    _game_log_handlers.clear()
    _endgame_logged.clear()
    _round_start_logged.clear()
    _active_game_log["key"] = None
    _active_game_log["handler"] = None


# -----------------------------------------
# Coordinate-orientation self-calibration
# -----------------------------------------
# The local simulator documents N=y-1 / E=x+1, but the live competition server
# has been observed to INVERT the N/S axis (a commanded "N" increases y), which
# left the bot steering away from every target and oscillating between two cells
# forever (never reaching an asteroid, so it never mined, recharged or jumped).
# Rather than hard-code either convention, we LEARN the real direction->delta
# mapping by watching the result of our own MOVEs and then steer with the
# learned signs. Defaults match the documented convention; a single observed
# move corrects a flipped axis. Orientation is a server-wide constant so it is
# kept across games; the last-move record is per game+round.
_axis = {"ns": -1, "ew": 1}        # ns: dy produced by "N"; ew: dx produced by "E"
_last_move = {"key": None, "direction": None, "frm": None}
# Deadlock tracking: consecutive ticks where our last action FAILED and left
# our position AND energy unchanged (the server keeps rejecting whatever we
# submit -- e.g. ~588 refused RECHARGEs in game 37102 round 2). Per game+round.
_stuck = {"key": None, "cell": None, "energy": None, "count": 0}


def _reset_navigation_state():
    """Forget learned orientation and last move (used by tests for isolation)."""
    _axis["ns"] = -1
    _axis["ew"] = 1
    _last_move["key"] = None
    _last_move["direction"] = None
    _last_move["frm"] = None
    _stuck["key"] = None
    _stuck["cell"] = None
    _stuck["energy"] = None
    _stuck["count"] = 0


def _update_stuck_state(ctx):
    """Count consecutive no-progress ticks and stash the run length on ``ctx``.

    A tick counts as "stuck" when our previous action was REJECTED (outcome
    FAILURE) yet our location and energy are identical to the prior tick -- i.e.
    the server keeps refusing what we send and we are making zero progress.
    Resets whenever anything changes or the action succeeds. The strategy reads
    ``ctx.stuck_count`` to decide when to force an escape action.
    """
    key = (ctx.game_id, ctx.round)
    no_change = (
        _stuck["key"] == key
        and _stuck["cell"] == ctx.location
        and _stuck["energy"] == ctx.energy
    )
    if no_change and ctx.last_action_failed:
        _stuck["count"] += 1
    else:
        _stuck["count"] = 0
    _stuck["key"] = key
    _stuck["cell"] = ctx.location
    _stuck["energy"] = ctx.energy
    ctx.stuck_count = _stuck["count"]
    return _stuck["count"]


def _calibrate_from_last_move(ctx):
    """Learn the real axis orientation from the result of our previous MOVE.

    Returns the cell we actually moved from this tick (our true previous cell)
    for anti-oscillation, or None when no usable prior move is known. Updates
    the module-level ``_axis`` map in place when an observed move contradicts the
    current mapping (e.g. a commanded "N" that increased y).
    """
    key = (ctx.game_id, ctx.round)
    direction = _last_move["direction"]
    frm = _last_move["frm"]
    if _last_move["key"] != key or direction is None or frm is None:
        return None
    cur = ctx.location
    if cur == frm:
        # No net movement. If our last MOVE was explicitly REJECTED while we
        # sat on a map edge, the server pushed us off-map: its sign for that
        # axis is the OPPOSITE of ours (e.g. we commanded "S" expecting y+1 from
        # y=0, but the server's "S" is y-1). Flip so we steer back inland next
        # tick. This is the only way to recalibrate at a boundary, where a wrong
        # axis otherwise pins us against the edge forever (the delta-based
        # learning below can never fire because the position never changes).
        if ctx.last_action_type == "MOVE" and ctx.last_action_failed:
            x, y = cur
            if direction in ("N", "S") and y in (0, ctx.map_height - 1):
                _axis["ns"] = -_axis["ns"]
                logger.debug(
                    "CALIBRATION ns flip -> %+d (MOVE %s rejected at y-edge %s)",
                    _axis["ns"], direction, cur,
                )
            elif direction in ("E", "W") and x in (0, ctx.map_width - 1):
                _axis["ew"] = -_axis["ew"]
                logger.debug(
                    "CALIBRATION ew flip -> %+d (MOVE %s rejected at x-edge %s)",
                    _axis["ew"], direction, cur,
                )
        return None
    dx, dy = cur[0] - frm[0], cur[1] - frm[1]
    if direction in ("N", "S") and dy != 0:
        observed_ns = dy if direction == "N" else -dy
        new_ns = 1 if observed_ns > 0 else -1
        if new_ns != _axis["ns"]:
            logger.debug(
                "CALIBRATION ns %s -> %s (commanded %s from %s, now at %s)",
                _axis["ns"], new_ns, direction, frm, cur,
            )
            _axis["ns"] = new_ns
    elif direction in ("E", "W") and dx != 0:
        observed_ew = dx if direction == "E" else -dx
        new_ew = 1 if observed_ew > 0 else -1
        if new_ew != _axis["ew"]:
            logger.debug(
                "CALIBRATION ew %s -> %s (commanded %s from %s, now at %s)",
                _axis["ew"], new_ew, direction, frm, cur,
            )
            _axis["ew"] = new_ew
    return frm


def _record_move(ctx, action):
    """Remember a just-emitted MOVE so the next tick can calibrate + reconstruct."""
    _last_move["key"] = (ctx.game_id, ctx.round)
    if action.get("actionType") == "MOVE":
        _last_move["direction"] = action.get("payload", {}).get("direction")
        _last_move["frm"] = ctx.location
    else:
        _last_move["direction"] = None
        _last_move["frm"] = None



class Tunables:
    """Decision thresholds. Defaults mirror the reference heuristic."""

    RECHARGE_LOW = 20            # start recharging below this absolute energy
    RECHARGE_HIGH_FRAC = 0.80    # stop recharging above this fraction of maxEnergy
    SELL_CARGO_THRESHOLD = 10    # cargo worth hauling to a (possibly distant) post
    PLUNDER_THRESHOLD = 5        # min stolen cargo worth a PLUNDER
    # Margins for committing to a fight: we must clearly out-gun the target on
    # BOTH offence (can we break them) and defence (can we take their hits).
    ATTACK_OFFENSE_MARGIN = 1.3
    ATTACK_DEFENSE_MARGIN = 1.3
    ATTACK_THREAT_FRACTION = 0.5
    COMPETITION_WEIGHT = 1.0     # penalty per competing miner on an asteroid
    # Only deploy panels to bank energy for a JUMP when the travel target is at
    # least this far. Below it, free MOVEs reach the target faster than banking
    # a flat ~jumpMinCost would (banking pays off only over long distances).
    # Used as the FLOOR for the per-map adaptive distance computed in
    # GameContext (slow recharge / pricey jumps push the real threshold higher).
    JUMP_BANK_MIN_DISTANCE = 12
    # Break a deadlock: if our last action is REJECTED with no change in
    # position or energy this many ticks in a row, the server keeps refusing
    # whatever we submit -- stop re-issuing it and force a cheap MOVE to dislodge.
    STUCK_ESCAPE_THRESHOLD = 3


# ---------------------------------------------------------
# Reusable utility classes
# ---------------------------------------------------------
class ShipUtils:
    """Stateless geometry and parsing helpers shared across strategies."""

    _OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

    @staticmethod
    def safe_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def location(entity):
        loc = entity.get("location", {}) if isinstance(entity, dict) else {}
        return int(loc.get("x", 0)), int(loc.get("y", 0))

    @staticmethod
    def distance(a, b):
        return math.dist(a, b)

    @staticmethod
    def chebyshev(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @classmethod
    def nearest(cls, origin, entities):
        if not entities:
            return None
        # Deterministic tie-break on coordinates so equidistant targets never
        # flip between turns (which would make the ship oscillate).
        return min(
            entities,
            key=lambda e: (cls.distance(origin, cls.location(e)),) + cls.location(e),
        )

    @staticmethod
    def opposite(direction):
        return ShipUtils._OPPOSITE.get(direction)

    @staticmethod
    def next_position(origin, direction):
        x, y = origin
        if direction == "N":
            return x, y + _axis["ns"]
        if direction == "S":
            return x, y - _axis["ns"]
        if direction == "E":
            return x + _axis["ew"], y
        if direction == "W":
            return x - _axis["ew"], y
        return x, y

    @staticmethod
    def move_dominant_axis(origin, target):
        ox, oy = origin
        tx, ty = target
        dx, dy = tx - ox, ty - oy
        if abs(dx) > abs(dy):
            return "E" if dx * _axis["ew"] > 0 else "W"
        if dy != 0:
            return "N" if dy * _axis["ns"] > 0 else "S"
        return None


class CombatEvaluator:
    """Estimates attack / defence power from observable ship fields.

    Enemy skills/stats are stripped on sensors (we only see a ReducedContactShip),
    so combat ability is proxied from ``skillPointsSpent``. Our own skills are
    known exactly. Grounded in the engine's combat-score weights:
        offence ~ attack_power*10 + attack_accuracy*5 + energy*0.5
        defence ~ health + shield_strength*8 + evade*3 (+ shield pool)
    """

    def __init__(self, attack_cost, max_energy):
        self.attack_cost = attack_cost
        self.max_energy = max_energy

    def skill_offense(self, skills, points):
        if skills:
            return (
                ShipUtils.safe_int(skills.get("attack_power")) * 10.0
                + ShipUtils.safe_int(skills.get("attack_accuracy")) * 5.0
            )
        # Skills hidden: assume points could be invested in offence (worst case).
        return points * 12.0

    def skill_defense(self, skills, points):
        if skills:
            return (
                ShipUtils.safe_int(skills.get("shield_strength")) * 8.0
                + ShipUtils.safe_int(skills.get("evade")) * 3.0
            )
        return points * 8.0

    def rough_power(self, ship):
        """Cheap overall power proxy used for flee/stand thresholds."""
        hp = ShipUtils.safe_int(ship.get("health"))
        en = ShipUtils.safe_int(ship.get("energy"))
        shield_val = ShipUtils.safe_int((ship.get("shield", {}) or {}).get("value"))
        skill_pts = ShipUtils.safe_int(ship.get("skillPointsSpent"))
        return hp * 1.0 + en * 1.0 + shield_val * 0.5 + skill_pts * 25.0

    def attack_power(self, ship, *, recharging=None, energy=None, skills=None, latent=False):
        """Estimated damage the ship can deal this turn.

        ``latent`` considers what they could do after retracting solar panels.
        Pass ``skills`` for our own ship (known); omit for enemies (proxied).
        """
        rech = bool(ship.get("recharging")) if recharging is None else recharging
        en = ShipUtils.safe_int(ship.get("energy")) if energy is None else energy
        if not latent and (rech or en < self.attack_cost):
            return 0.0
        base = (
            self.skill_offense(skills, ShipUtils.safe_int(ship.get("skillPointsSpent")))
            + min(en, self.max_energy) * 0.5
        )
        if latent and rech:
            base *= 0.5  # delayed: must end recharge before it can attack
        return base

    def survivability(self, ship, *, recharging=None, shields_up=None, skills=None):
        """Estimated ability to absorb incoming damage."""
        rech = bool(ship.get("recharging")) if recharging is None else recharging
        shield = ship.get("shield", {}) or {}
        powered = (
            str(shield.get("state", "")).upper() == "POWERED"
            if shields_up is None
            else shields_up
        )
        base = (
            ShipUtils.safe_int(ship.get("health")) * 1.0
            + self.skill_defense(skills, ShipUtils.safe_int(ship.get("skillPointsSpent")))
        )
        if powered:
            # Shields up: ~25% incoming reduction + the shield pool itself.
            base = base * 1.33 + ShipUtils.safe_int(shield.get("value")) * 0.5
        if rech:
            base *= 0.5  # panels out: cannot raise shields, combat-penalised
        return base


# -------------------------------------------------
# Parsed game state
# -------------------------------------------------
class GameContext:
    """Parses one ActionRequest into structured, reusable state."""

    def __init__(self, action_request, now_ms=None):
        self.request = action_request or {}
        self.me = self.request.get("me", {}) or {}
        self.contacts = self.request.get("sensors", []) or []
        self.game_state = self.request.get("gameState", {}) or {}
        self.metadata = self.game_state.get("metadata", {}) or {}

        ship_config = self.metadata.get("shipConfig", {}) or {}
        energy_costs = ship_config.get("energyCosts", {}) or {}
        map_config = self.metadata.get("mapConfig", {}) or {}
        self.action_restrictions = self.metadata.get("actionRestrictions", {}) or {}

        # --- Config / costs (real game metadata) ---
        self.jump_cost_per_unit = float(energy_costs.get("jump", 1))
        # jumpMinCost: minimum energy any jump costs. Default to the observed
        # live value (75) rather than 0 so we never emit unaffordable jumps.
        self.jump_min_cost = int(energy_costs.get("jumpMinCost", 75))
        self.move_cost = int(energy_costs.get("move", 0))
        self.mine_cost = int(energy_costs.get("mine", 10))
        self.plunder_cost = int(energy_costs.get("plunder", 5))
        self.negotiate_cost = int(energy_costs.get("negotiate", 5))
        # ATTACK cost is not published; the engine treats it as a minimum of 1.
        self.attack_cost = int(energy_costs.get("attack", 1))
        self.max_jump_distance = int(ship_config.get("maxJumpDistance", 50))
        self.max_energy = int(ship_config.get("maxEnergy", 100))
        # Energy regained per RECHARGE tick. Drives the adaptive bank distance:
        # a slow recharge means we crawl many cells while charging, so banking
        # only pays off for farther targets.
        self.energy_per_recharge = int(ship_config.get("energyPerRecharge", 10)) or 10
        # Distance beyond which banking energy for a JUMP beats crawling. We
        # crawl 1 free cell/tick while charging energy_per_recharge/tick, so the
        # minimum jump (jump_min_cost energy) is banked after roughly
        # jump_min_cost/energy_per_recharge ticks -- and during those ticks we
        # already advance that many cells for free. Banking only buys time when
        # the target is farther than that crawl, with the static floor as a
        # lower bound. Adapts to tight-energy maps (high jump cost / slow
        # recharge, e.g. game 37102: jump/u=2.0, recharge=8 -> 160x160 map).
        self.jump_bank_min_distance = max(
            Tunables.JUMP_BANK_MIN_DISTANCE,
            int(math.ceil(self.jump_min_cost / self.energy_per_recharge)),
        )
        self.map_width = int(map_config.get("width", 125))
        self.map_height = int(map_config.get("height", 125))

        mining_cfg = self.metadata.get("mining", {}) or {}
        self.mine_base_success = float(mining_cfg.get("baseSuccessChance", 0.5))
        self.mine_min_payout = int(mining_cfg.get("minPayout", 1))
        self.mine_max_payout = int(mining_cfg.get("maxPayout", 10))

        # --- Ship state ---
        self.state = str(self.me.get("state", "")).upper()
        self.energy = ShipUtils.safe_int(self.me.get("energy"))
        self.nutrinium = ShipUtils.safe_int(self.me.get("nutrinium"))
        self.health = ShipUtils.safe_int(self.me.get("health"), 100)
        self.recharging = bool(self.me.get("recharging", False))
        self.location = ShipUtils.location(self.me)
        self.player_id = self.me.get("playerId")
        self.team_id = self.me.get("teamId")
        self.skills = self.me.get("skills", {}) or {}
        self.modules = self.me.get("modules", []) or []
        self.has_jump = "JUMP" in self.modules
        self.shields_up = str((self.me.get("shield", {}) or {}).get("state", "DOWN")).upper() == "POWERED"

        objectives = self.me.get("objectives", {}) or {}
        negotiate_objective = objectives.get("negotiate", {}) or {}
        self.objective_post_name = negotiate_objective.get("tradingPostName")

        # --- Sensor groups ---
        self.trading_posts = [c for c in self.contacts if c.get("type") == "trading_post"]
        self.asteroids = [c for c in self.contacts if c.get("type") == "asteroid"]
        self.wreckage = [c for c in self.contacts if c.get("type") == "wreckage"]
        self.mineable_asteroids = [
            a for a in self.asteroids if ShipUtils.safe_int(a.get("nutrinium")) > 0
        ]
        self.enemy_ships = [c for c in self.contacts if self._is_enemy(c)]
        self.same_zone_enemies = [
            e for e in self.enemy_ships if ShipUtils.location(e) == self.location
        ]
        # Other live ships (enemies AND teammates) for contention scoring.
        self.ship_contacts = [
            c for c in self.contacts
            if c.get("type") == "ship"
            and str(c.get("state", "")).upper() != "DESTROYED"
            and c.get("playerId") != self.player_id
        ]

        self.at_trading_post = any(
            ShipUtils.location(tp) == self.location for tp in self.trading_posts
        )
        self.at_asteroid = any(
            ShipUtils.location(a) == self.location and ShipUtils.safe_int(a.get("nutrinium")) > 0
            for a in self.asteroids
        )
        self.objective_post = next(
            (tp for tp in self.trading_posts if tp.get("name") == self.objective_post_name),
            None,
        )
        self.at_objective_post = (
            self.objective_post is not None
            and ShipUtils.location(self.objective_post) == self.location
        )

        # --- Round clock (end-game banking) ---
        self.ticks_remaining = self._estimate_ticks_remaining(now_ms)

        # --- Movement history (anti-oscillation) ---
        self.prev_cell_source = "none"
        self.prev_cell = self._previous_cell()
        if self.prev_cell == self.location:
            self.prev_cell = None
            self.prev_cell_source = "none"

        # --- Last-action outcome (avoid re-emitting a just-failed action) ---
        # PLUNDER/ATTACK target a same-zone enemy from a snapshot that is one
        # tick stale: the enemy may have moved out of the zone before the action
        # resolved ("the target is no longer in the zone"). When our previous
        # offence action FAILED we back off offence for a tick and do productive
        # work instead, so a mobile enemy passing through cannot trap us in a
        # fail-retry loop (retrying ATTACK after a failed PLUNDER -- or vice
        # versa -- hits the same absent target).
        last_result = self.request.get("actionResult", {}) or {}
        self.last_action_type = str(last_result.get("actionType", "")).upper()
        self.last_action_failed = (
            str(last_result.get("outcome", "")).upper() == "FAILURE"
        )
        # Why an action was accepted/rejected: the result payload carries a
        # machine resultCode (e.g. "RECHARGE_FAILURE") plus a human message.
        # Surfacing these makes a server refusal (such as the ~588 rejected
        # RECHARGEs in game 37102 round 2) visible instead of an opaque FAIL.
        last_payload = last_result.get("payload", {}) or {}
        self.last_action_result_code = str(last_payload.get("resultCode", "") or "")
        self.last_action_message = str(last_payload.get("message", "") or "")
        self.offence_target_lost = (
            self.last_action_failed and self.last_action_type in ("PLUNDER", "ATTACK")
        )
        # Consecutive no-progress ticks (filled per tick by _update_stuck_state).
        self.stuck_count = 0

        # --- Scoreboard / outcome fields (end-of-game reporting) ---
        self.game_id = self.game_state.get("gameId", self.me.get("gameId"))
        self.tick = ShipUtils.safe_int(self.game_state.get("tick"))
        self.round = ShipUtils.safe_int(self.game_state.get("round"))
        self.credits = ShipUtils.safe_int(self.me.get("credits"))
        self.stats = self.me.get("stats", {}) or {}
        self.round_scores = self.me.get("roundScores", []) or []
        self.leaderboard = self.request.get("leaderboard", []) or []
        self.skill_points_spent = ShipUtils.safe_int(self.me.get("skillPointsSpent"))
        self.skill_points_total = ShipUtils.safe_int(self.me.get("skillPointsTotal"))

        # Extra fields the shared action-masker safety net validates against.
        self.shields_cost = int(energy_costs.get("shields", 5))
        self.salvage_energy_cost = int(energy_costs.get("salvage", 999))
        self.max_health = int(ship_config.get("maxHealth", 100))
        shield = self.me.get("shield", {}) or {}
        self.shield_state = str(shield.get("state", "DOWN")).upper()
        self.shield_value = ShipUtils.safe_int(shield.get("value", shield.get("strength", 0)))
        self.shield_capacity = ShipUtils.safe_int(shield.get("capacity", shield.get("maxStrength", 0)))

        self.combat = CombatEvaluator(self.attack_cost, self.max_energy)

    def my_leaderboard_entry(self):
        """My row in the leaderboard (carries rank/position + gameScore), or None."""
        for entry in self.leaderboard:
            if entry.get("playerId") == self.player_id:
                return entry
        return None

    # --- Identity / classification ---
    def _is_enemy(self, contact):
        if contact.get("type") != "ship":
            return False
        if str(contact.get("state", "")).upper() == "DESTROYED":
            return False
        contact_team = contact.get("teamId")
        if self.team_id is not None and contact_team is not None:
            return contact_team != self.team_id
        return contact.get("playerId") != self.player_id

    def miners_at(self, loc):
        return sum(1 for s in self.ship_contacts if ShipUtils.location(s) == loc)

    def in_bounds(self, x, y):
        return 0 <= x < self.map_width and 0 <= y < self.map_height

    # --- Round clock estimation ---
    def _estimate_ticks_remaining(self, now_ms):
        start = ShipUtils.safe_int(self.game_state.get("start"))
        end = ShipUtils.safe_int(self.game_state.get("end"))
        tick = ShipUtils.safe_int(self.game_state.get("tick"))
        now = int(time.time() * 1000) if now_ms is None else now_ms
        # Only trust a genuinely live round window (replayed fixtures are stale).
        live = end > start and start <= now <= end
        if live and tick > 0:
            elapsed = max(now - start, 1)
            ms_per_tick = elapsed / tick
            return (end - now) / ms_per_tick
        return None

    # --- Reconstruct our previous cell to damp two-cell oscillation ---
    def _cell_before_move(self, action_type, outcome, payload):
        if str(action_type or "").upper() != "MOVE":
            return None
        if str(outcome or "SUCCESS").upper() != "SUCCESS":
            return None
        payload = payload or {}
        frm = payload.get("from")
        if isinstance(frm, (list, tuple)) and len(frm) == 2:
            return (ShipUtils.safe_int(frm[0]), ShipUtils.safe_int(frm[1]))
        if isinstance(frm, dict) and "x" in frm:
            return (ShipUtils.safe_int(frm.get("x")), ShipUtils.safe_int(frm.get("y")))
        direction = payload.get("direction")
        if direction in ("N", "S", "E", "W"):
            return ShipUtils.next_position(self.location, ShipUtils.opposite(direction))
        return None

    def _previous_cell(self):
        last_result = self.request.get("actionResult", {}) or {}
        cell = self._cell_before_move(
            last_result.get("actionType"),
            last_result.get("outcome", "SUCCESS"),
            last_result.get("payload", {}),
        )
        if cell is not None:
            self.prev_cell_source = "actionResult"
            return cell
        for evt in reversed(self.request.get("eventLog", []) or []):
            if evt.get("playerId") != self.player_id:
                continue
            cell = self._cell_before_move(
                evt.get("actionType"), evt.get("outcome", "SUCCESS"), evt.get("payload", {})
            )
            if cell is not None:
                self.prev_cell_source = "eventLog"
                return cell
            break
        self.prev_cell_source = "none"
        return None

    # --- Restriction helpers (driven by round metadata) ---
    def allowed_while_recharging(self, action_type):
        return self.action_restrictions.get(action_type, {}).get("allowedWhileRecharging", True)

    def allowed_with_shields_up(self, action_type):
        return self.action_restrictions.get(action_type, {}).get("allowedWithShieldsUp", True)

    # --- Jump economics ---
    def jump_energy(self, dist):
        if dist <= 0:
            return 0
        return max(self.jump_min_cost, int(math.ceil(dist * self.jump_cost_per_unit)))

    def can_jump(self, dist):
        return (
            self.has_jump
            and 0 < dist <= self.max_jump_distance
            and self.energy >= self.jump_energy(dist)
        )

    # --- Asteroid value model ---
    def asteroid_score(self, asteroid):
        nutr = ShipUtils.safe_int(asteroid.get("nutrinium"))
        if nutr <= 0:
            return 0.0
        mass = max(ShipUtils.safe_int(asteroid.get("mass"), 1), 1)
        density = nutr / mass
        success_chance = min(1.0, density * 10.0 * self.mine_base_success)
        expected_payout = (self.mine_min_payout + min(self.mine_max_payout, nutr)) / 2.0
        expected_yield = success_chance * expected_payout
        pool_factor = 1.0 + math.log1p(nutr)
        dist = ShipUtils.distance(self.location, ShipUtils.location(asteroid))
        competition_factor = 1.0 + Tunables.COMPETITION_WEIGHT * self.miners_at(
            ShipUtils.location(asteroid)
        )
        return (expected_yield * pool_factor) / ((dist + 1.0) * competition_factor)


# -----------------------------------
# Heuristic strategy
# -----------------------------------
class HeuristicStrategy:
    """Priority-ordered decision policy with an action-mask safety net."""

    def __init__(self, ctx):
        self.ctx = ctx
        # Set by _jump_waypoint when a jump toward the chosen travel target is
        # wanted but unaffordable AND the target is far enough that banking
        # energy to jump beats crawling. Read by _choose to deploy panels.
        self.bank_for_jump = False
        self._compute_threats()

    # --- Threat model ---
    def _compute_threats(self):
        ctx = self.ctx
        combat = ctx.combat
        self.my_power = combat.rough_power(ctx.me)
        # Panels out = weak: cannot ATTACK / RAISE_SHIELDS, combat-penalised.
        self.my_power_eff = self.my_power * (0.5 if ctx.recharging else 1.0)

        def can_act(enemy):
            return ShipUtils.safe_int(enemy.get("energy")) > 0

        def threat_dist(enemy):
            return ShipUtils.chebyshev(ShipUtils.location(enemy), ctx.location)

        # Every enemy that can still act this turn (has energy). Used to judge
        # whether a navigation *destination* is contested -- not just the
        # overwhelming "dominant" raiders we actively flee.
        self.acting_enemies = [e for e in ctx.enemy_ships if can_act(e)]
        self.threats = [e for e in self.acting_enemies if threat_dist(e) <= 1]
        threat_power = sum(combat.rough_power(e) for e in self.threats)
        strongest = max((combat.rough_power(e) for e in self.threats), default=0.0)

        cargo_risk = min(ctx.nutrinium / 50.0, 1.0)
        health_risk = 1.0 - min(ctx.health / 100.0, 1.0)
        risk = max(cargo_risk, health_risk)
        overwhelm_ratio = 2.0 - 1.0 * risk  # cautious (2x) .. eager to flee (1x)

        self.is_overwhelmed = bool(self.threats) and (
            threat_power >= overwhelm_ratio * max(self.my_power_eff, 1.0)
            or strongest >= overwhelm_ratio * max(self.my_power_eff, 1.0)
        )
        self.dominant_threats = [
            e for e in ctx.enemy_ships
            if can_act(e)
            and combat.rough_power(e) >= overwhelm_ratio * max(self.my_power_eff, 1.0)
        ]
        # Drop rocks guarded by a dominant raider so we never path toward them.
        if self.dominant_threats:
            ctx.mineable_asteroids = [
                a for a in ctx.mineable_asteroids if not self._rock_guarded(a)
            ]

    def _is_unsafe(self, cell):
        for enemy in self.dominant_threats:
            if ShipUtils.chebyshev(ShipUtils.location(enemy), cell) <= 1:
                return True
        return False

    def _rock_guarded(self, asteroid):
        ax_ay = ShipUtils.location(asteroid)
        for enemy in self.dominant_threats:
            if ShipUtils.chebyshev(ShipUtils.location(enemy), ax_ay) <= 2:
                return True
        return False

    def _threat_at_destination(self, target_xy):
        """True if any enemy that can act sits on (or adjacent to) the target.

        Arriving on a contested cell is dangerous: a JUMP resolves last, so we
        would land next to the enemy and take their hit; a MOVE walks straight
        into their attack range. We treat Chebyshev <= 1 of the destination as
        contested so navigation can prefer an uncontested target instead.
        """
        cell = (int(target_xy[0]), int(target_xy[1]))
        for enemy in self.acting_enemies:
            if ShipUtils.chebyshev(ShipUtils.location(enemy), cell) <= 1:
                return True
        return False

    # --- Action constructors ---
    def _make_move(self, direction):
        nx, ny = ShipUtils.next_position(self.ctx.location, direction)
        if not self.ctx.in_bounds(nx, ny):
            return {"actionType": "WAIT"}
        return {"actionType": "MOVE", "payload": {"direction": direction}}

    def _make_jump(self, tx, ty):
        cx = min(max(0, int(tx)), self.ctx.map_width - 1)
        cy = min(max(0, int(ty)), self.ctx.map_height - 1)
        if (cx, cy) == self.ctx.location:
            return {"actionType": "WAIT"}
        return {"actionType": "JUMP", "payload": {"target_location": {"x": cx, "y": cy}}}

    def _escape_action(self):
        """Force a fresh in-bounds MOVE to break a server-rejection deadlock.

        Called only when ``ctx.stuck_count`` shows our action has been refused
        repeatedly with no change in position or energy. We rotate the tried
        direction by the stuck count so successive escape attempts probe
        different headings, guaranteeing forward motion (MOVE is free) that
        dislodges whatever the server keeps rejecting. Returns ``None`` only if
        no neighbouring cell is in bounds (impossible on a >1x1 map).
        """
        ctx = self.ctx
        order = ["N", "E", "S", "W"]
        start = ctx.stuck_count % len(order)
        for i in range(len(order)):
            direction = order[(start + i) % len(order)]
            nx, ny = ShipUtils.next_position(ctx.location, direction)
            if ctx.in_bounds(nx, ny) and (nx, ny) != ctx.location:
                logger.debug(
                    "ESCAPE deadlock: stuck=%d last=%s/%s reason=%s/%s -> MOVE %s "
                    "from %s to %s",
                    ctx.stuck_count, ctx.last_action_type or "-",
                    "FAIL" if ctx.last_action_failed else "ok",
                    ctx.last_action_result_code or "-",
                    ctx.last_action_message or "-",
                    direction, ctx.location, (nx, ny),
                )
                return {"actionType": "MOVE", "payload": {"direction": direction}}
        return None

    # --- Movement helpers (threat-aware) ---
    def _safe_move_dir(self, target_xy):
        ctx = self.ctx
        ox, oy = ctx.location
        tx, ty = target_xy
        dx, dy = tx - ox, ty - oy
        horiz = ("E" if dx * _axis["ew"] > 0 else "W") if dx != 0 else None
        vert = ("N" if dy * _axis["ns"] > 0 else "S") if dy != 0 else None
        order = [horiz, vert] if abs(dx) >= abs(dy) else [vert, horiz]
        rejected = []
        for direction in order:
            if direction is None:
                continue
            nx, ny = ShipUtils.next_position(ctx.location, direction)
            if not ctx.in_bounds(nx, ny):
                rejected.append((direction, "out-of-bounds"))
                continue
            if self._is_unsafe((nx, ny)):
                rejected.append((direction, "unsafe"))
                continue
            logger.debug(
                "NAV _safe_move_dir from %s -> target %s: dx=%d dy=%d order=%s "
                "chose %s (rejected=%s)",
                ctx.location, (tx, ty), dx, dy, order, direction, rejected,
            )
            return direction
        logger.debug(
            "NAV _safe_move_dir from %s -> target %s: dx=%d dy=%d order=%s "
            "NO safe direction (rejected=%s)",
            ctx.location, (tx, ty), dx, dy, order, rejected,
        )
        return None

    def _travel_towards(self, target_xy):
        ctx = self.ctx
        # Prefer a JUMP to cover ground fast whenever one is possible. A jump
        # costs a flat minimum (jumpMinCost) regardless of distance, so once we
        # commit we hop as far toward the target as we can: directly onto it
        # when in range, otherwise to the farthest safe waypoint along the path
        # (targets beyond maxJumpDistance). Falls back to a free MOVE step.
        # Reset here so the flag reflects only this (the chosen) travel target:
        # _work_action returns on the first travel that yields an action, so the
        # last _jump_waypoint call is the one that matters.
        self.bank_for_jump = False
        jump_target = self._jump_waypoint(target_xy)
        if jump_target is not None:
            return self._make_jump(jump_target[0], jump_target[1])
        direction = self._safe_move_dir(target_xy)
        if direction and ctx.energy >= ctx.move_cost:
            return self._make_move(direction)
        return None

    def _jump_waypoint(self, target_xy):
        """Cell to JUMP to when heading for target_xy, or None to MOVE instead.

        Returns None for adjacent targets (a free MOVE is cheaper than a flat
        ~jumpMinCost jump), when JUMP is locked/unaffordable, or when no safe
        landing exists. When the target is within range and safe we jump
        straight onto it. When the target itself is contested/dangerous -- or
        farther than maxJumpDistance -- we hop to the farthest SAFE cell along
        the straight line toward it instead of crawling the whole way one grid
        at a time. Jumping onto a contested cell is still avoided (JUMP resolves
        last, so we would land beside a raider and eat a free hit), but stopping
        a couple of cells short of the danger covers ground fast without that
        risk.
        """
        ctx = self.ctx
        if not ctx.has_jump:
            logger.debug("JUMP skip: no JUMP module unlocked (modules=%s)", ctx.modules)
            return None
        ox, oy = ctx.location
        tx, ty = int(target_xy[0]), int(target_xy[1])
        dist = ShipUtils.distance(ctx.location, target_xy)
        if dist <= 1:
            logger.debug(
                "JUMP skip: target %s is adjacent (dist=%.2f); free MOVE is cheaper",
                (tx, ty), dist,
            )
            return None
        # In range and safe: jump straight onto the target.
        if dist <= ctx.max_jump_distance:
            affordable = ctx.can_jump(dist)
            unsafe = self._is_unsafe((tx, ty))
            threatened = self._threat_at_destination(target_xy)
            if affordable and not unsafe and not threatened:
                logger.debug(
                    "JUMP direct: %s -> %s dist=%.2f cost=%d energy=%d",
                    ctx.location, (tx, ty), dist, ctx.jump_energy(dist), ctx.energy,
                )
                return (tx, ty)
            if not affordable:
                if dist >= ctx.jump_bank_min_distance:
                    # Far enough that banking energy to jump beats crawling.
                    self.bank_for_jump = True
                logger.debug(
                    "JUMP skip (in range): target=%s dist=%.2f need_energy=%d "
                    "have=%d bank=%s -> MOVE one grid",
                    (tx, ty), dist, ctx.jump_energy(dist), ctx.energy,
                    self.bank_for_jump,
                )
                return None
            # Affordable but the destination itself is contested/unsafe. Rather
            # than crawl the entire way into the danger zone, fall through and
            # jump to the farthest SAFE cell short of it.
            logger.debug(
                "JUMP in range but target contested: target=%s dist=%.2f "
                "unsafe=%s threatened=%s -> seeking safe waypoint short of it",
                (tx, ty), dist, unsafe, threatened,
            )
        # Hop to the farthest safe, affordable cell along the line to target.
        # Covers both out-of-range targets and in-range contested ones. We stop
        # at a 2-cell minimum hop: paying a flat ~jumpMinCost to advance a
        # single grid (when MOVE is free) is never worth it.
        max_reach = min(ctx.max_jump_distance, int(math.ceil(dist)))
        ux, uy = (tx - ox) / dist, (ty - oy) / dist
        rejected = []
        for reach in range(max_reach, 1, -1):
            cx = min(max(0, int(round(ox + ux * reach))), ctx.map_width - 1)
            cy = min(max(0, int(round(oy + uy * reach))), ctx.map_height - 1)
            if (cx, cy) == (ox, oy) or (cx, cy) == (tx, ty):
                continue
            hop = ShipUtils.distance(ctx.location, (cx, cy))
            if hop < 2:
                continue
            if not ctx.can_jump(hop):
                rejected.append((reach, "unaffordable", hop))
                continue
            if self._is_unsafe((cx, cy)) or self._threat_at_destination((cx, cy)):
                rejected.append((reach, "blocked", hop))
                continue
            logger.debug(
                "JUMP waypoint: %s -> %s (reach=%d hop=%.2f cost=%d energy=%d) "
                "toward target %s dist=%.2f",
                ctx.location, (cx, cy), reach, hop, ctx.jump_energy(hop),
                ctx.energy, (tx, ty), dist,
            )
            return (cx, cy)
        if ctx.energy < ctx.jump_min_cost:
            # The far target's hops were unaffordable: bank energy to jump (the
            # distance always exceeds jump_bank_min_distance here).
            self.bank_for_jump = True
        logger.debug(
            "JUMP skip (target %s dist=%.2f): no reachable/safe waypoint "
            "energy=%d need>=%d bank=%s rejected=%s -> MOVE one grid",
            (tx, ty), dist, ctx.energy, ctx.jump_min_cost, self.bank_for_jump,
            rejected,
        )
        return None

    def _flee_action(self):
        ctx = self.ctx
        if not self.threats:
            return None
        tx = sum(ShipUtils.location(e)[0] for e in self.threats) / len(self.threats)
        ty = sum(ShipUtils.location(e)[1] for e in self.threats) / len(self.threats)
        nearest_post = ShipUtils.nearest(ctx.location, ctx.trading_posts)
        post_xy = ShipUtils.location(nearest_post) if nearest_post is not None else None

        best_dir, best_key = None, None
        for direction in ("N", "S", "E", "W"):
            nx, ny = ShipUtils.next_position(ctx.location, direction)
            if not ctx.in_bounds(nx, ny):
                continue
            away = (nx - tx) ** 2 + (ny - ty) ** 2
            toward_post = -ShipUtils.distance((nx, ny), post_xy) if post_xy else 0.0
            key = (
                0 if self._is_unsafe((nx, ny)) else 1,
                away,
                0 if (nx, ny) == ctx.prev_cell else 1,
                toward_post,
            )
            if best_key is None or key > best_key:
                best_key, best_dir = key, direction
        return self._make_move(best_dir) if best_dir is not None else None

    # --- Offence ---
    def _attack_opportunity(self):
        ctx = self.ctx
        if ctx.recharging or ctx.energy < ctx.attack_cost:
            return None
        # Back off for a tick if our last offence just failed (target left the
        # zone): the same enemy is likely still absent, so retrying only loops.
        if ctx.offence_target_lost:
            return None
        targets = [
            e for e in ctx.same_zone_enemies
            if str(e.get("state", "")).upper() != "DESTROYED" and e.get("playerId")
        ]
        if not targets:
            return None

        combat = ctx.combat
        my_atk = combat.attack_power(
            ctx.me, recharging=ctx.recharging, energy=ctx.energy, skills=ctx.skills
        )
        my_surv = max(
            combat.survivability(
                ctx.me, recharging=ctx.recharging, shields_up=ctx.shields_up, skills=ctx.skills
            ),
            1.0,
        )
        best = None
        for enemy in targets:
            e_surv = max(combat.survivability(enemy), 1.0)
            e_atk_now = combat.attack_power(enemy, latent=False)
            e_atk_latent = combat.attack_power(enemy, latent=True)
            cargo = ShipUtils.safe_int(enemy.get("nutrinium"))

            dominate = my_atk >= Tunables.ATTACK_OFFENSE_MARGIN * e_surv
            survive = my_surv >= Tunables.ATTACK_DEFENSE_MARGIN * max(e_atk_now, 1.0)
            worthwhile = (
                cargo >= Tunables.PLUNDER_THRESHOLD
                or e_atk_latent >= Tunables.ATTACK_THREAT_FRACTION * my_surv
            )
            if dominate and survive and worthwhile:
                key = (cargo, e_surv, enemy.get("playerId") or "")
                if best is None or key > best[0]:
                    best = (key, enemy)
        if best is None:
            return None
        enemy = best[1]
        e_hp = max(ShipUtils.safe_int(enemy.get("health"), 1), 1)
        needed = int(math.ceil(e_hp / 1.5)) + 2
        spend = max(ctx.attack_cost, min(ctx.energy, needed))
        return {"actionType": "ATTACK", "payload": {"target": enemy.get("playerId"), "energy": spend}}

    # --- End-game banking ---
    def _ticks_to_bank(self, post_xy):
        ctx = self.ctx
        dist = ShipUtils.distance(ctx.location, post_xy)
        if dist <= 0:
            return 1
        travel = 1 if (ctx.can_jump(dist) and dist <= ctx.max_jump_distance) else int(math.ceil(dist))
        return travel + 1

    def _endgame_bank_action(self):
        ctx = self.ctx
        if ctx.ticks_remaining is None or ctx.nutrinium <= 0:
            return None
        nearest_post = ShipUtils.nearest(ctx.location, ctx.trading_posts)
        if nearest_post is None:
            logger.debug(
                "ENDGAME no reachable post to bank nutrinium=%d (ticks_left=%.1f)",
                ctx.nutrinium, ctx.ticks_remaining,
            )
            return None
        post_xy = ShipUtils.location(nearest_post)
        if post_xy == ctx.location:
            logger.debug(
                "ENDGAME selling on post %s nutrinium=%d (ticks_left=%.1f)",
                post_xy, ctx.nutrinium, ctx.ticks_remaining,
            )
            return {"actionType": "SELL", "payload": {"nutrinium": ctx.nutrinium}}
        needed = self._ticks_to_bank(post_xy)
        # Commit to banking with a generous, distance-scaled cushion. The round
        # clock is a wall-clock estimate (no max-tick field) that can jump
        # between ticks, and real travel may burn more game-ticks than the
        # linear estimate (one MOVE per cell assumes we are polled every tick).
        # Banking a tick "too early" only costs a little mining; banking a tick
        # too late forfeits the whole cargo (~98 credits/unit), so we err early.
        safety = max(3.0, 0.35 * needed) + 1.0
        if ctx.ticks_remaining > needed + safety:
            logger.debug(
                "ENDGAME defer bank: nutrinium=%d post=%s dist=%.1f needed=%d "
                "safety=%.1f ticks_left=%.1f -> keep working",
                ctx.nutrinium, post_xy, ShipUtils.distance(ctx.location, post_xy),
                needed, safety, ctx.ticks_remaining,
            )
            return None
        logger.debug(
            "ENDGAME rush to bank: nutrinium=%d post=%s dist=%.1f needed=%d "
            "safety=%.1f ticks_left=%.1f -> head to post",
            ctx.nutrinium, post_xy, ShipUtils.distance(ctx.location, post_xy),
            needed, safety, ctx.ticks_remaining,
        )
        return self._travel_towards(post_xy)

    # --- Productive intent (ignores solar-panel state) ---
    def _work_action(self):
        ctx = self.ctx
        # Standing on a post with cargo: selling is free, always cash in.
        if ctx.at_trading_post and ctx.nutrinium > 0:
            return {"actionType": "SELL", "payload": {"nutrinium": ctx.nutrinium}}

        # Haul to a post once the load is worthwhile or nothing left to mine.
        if ctx.nutrinium > 0 and (
            ctx.nutrinium >= Tunables.SELL_CARGO_THRESHOLD or not ctx.mineable_asteroids
        ):
            nearest_post = ShipUtils.nearest(ctx.location, ctx.trading_posts)
            if nearest_post is not None:
                travel = self._travel_towards(ShipUtils.location(nearest_post))
                if travel is not None:
                    return travel

        # PLUNDER a same-zone enemy (keeps their cargo, unlike ATTACK).
        # Skip for one tick if our last offence just failed (target left the
        # zone): retrying immediately against a mobile enemy only loops on the
        # same failure -- do productive work (mine/move) instead.
        if ctx.energy >= ctx.plunder_cost and not ctx.shields_up and not ctx.offence_target_lost:
            plunder_targets = [
                e for e in ctx.same_zone_enemies
                if ShipUtils.safe_int(e.get("nutrinium")) >= Tunables.PLUNDER_THRESHOLD
            ]
            if plunder_targets:
                richest = max(
                    plunder_targets,
                    key=lambda e: (ShipUtils.safe_int(e.get("nutrinium")), e.get("playerId") or ""),
                )
                if richest.get("playerId"):
                    return {"actionType": "PLUNDER", "payload": {"target": richest["playerId"]}}

        # MINE the current asteroid while it still holds nutrinium.
        if ctx.at_asteroid and ctx.energy >= ctx.mine_cost:
            current = next(
                (a for a in ctx.asteroids
                 if ShipUtils.location(a) == ctx.location
                 and ShipUtils.safe_int(a.get("nutrinium")) > 0),
                None,
            )
            if current is not None:
                return {"actionType": "MINE"}

        # Travel to the best mineable asteroid we can approach safely. Rocks
        # with an enemy sitting on (or next to) them are pushed to the back of
        # the queue so we prefer an uncontested target, only diverting to a
        # threatened one when nothing safer is reachable.
        for ast in sorted(
            ctx.mineable_asteroids,
            key=lambda a: (
                1 if self._threat_at_destination(ShipUtils.location(a)) else 0,
                -ctx.asteroid_score(a),
            ) + ShipUtils.location(a),
        ):
            target = ShipUtils.location(ast)
            travel = self._travel_towards(target)
            if travel is not None:
                logger.debug(
                    "WORK travel to asteroid %s (nutrinium=%s score=%.3f) -> %s",
                    target,
                    ShipUtils.safe_int(ast.get("nutrinium")),
                    ctx.asteroid_score(ast),
                    travel.get("actionType"),
                )
                return travel

        # Nothing left to mine but still carrying cargo: go sell it.
        if ctx.nutrinium > 0 and not ctx.mineable_asteroids:
            nearest_post = ShipUtils.nearest(ctx.location, ctx.trading_posts)
            if nearest_post is not None:
                travel = self._travel_towards(ShipUtils.location(nearest_post))
                if travel is not None:
                    return travel

        # NEGOTIATE to build the team sell bonus when idle at the objective post.
        if ctx.at_objective_post and ctx.nutrinium == 0 and ctx.energy >= ctx.negotiate_cost:
            return {"actionType": "NEGOTIATE"}

        return {"actionType": "WAIT"}

    # --- Top-level decision (wraps work with panel management) ---
    def _choose(self):
        ctx = self.ctx
        if ctx.state == "DESTROYED":
            return {"actionType": "RESPAWN"}

        # DEADLOCK ESCAPE: the server has rejected our action for several ticks
        # running with zero change in position or energy (e.g. ~588 refused
        # RECHARGEs in game 37102 round 2). Re-submitting the same rejected
        # action will keep failing, so force a cheap MOVE to break out before
        # any other logic can re-pick it.
        if ctx.stuck_count >= Tunables.STUCK_ESCAPE_THRESHOLD:
            escape = self._escape_action()
            if escape is not None:
                return escape

        # END-GAME: bank cargo before the round clock runs out (takes priority).
        endgame = self._endgame_bank_action()
        if endgame is not None:
            return endgame

        # DEFENSE: flee an overwhelming nearby threat (banking first if on a post).
        if self.is_overwhelmed:
            if ctx.at_trading_post and ctx.nutrinium > 0 and not ctx.recharging:
                return {"actionType": "SELL", "payload": {"nutrinium": ctx.nutrinium}}
            flee = self._flee_action()
            if flee is not None:
                return flee

        # DEFENSIVE RETRACT: panels out next to a real threat is the worst footing.
        if ctx.recharging and self.threats:
            return {"actionType": "RECHARGE_END"}

        # ATTACK: press a clear advantage over an in-zone enemy.
        attack = self._attack_opportunity()
        if attack is not None:
            return attack

        work = self._work_action()
        work_type = work.get("actionType", "WAIT")

        if ctx.recharging:
            # MINE/MOVE/PLUNDER allowed while recharging -> keep working.
            if not ctx.allowed_while_recharging(work_type):
                return {"actionType": "RECHARGE_END"}
            if work_type == "WAIT" and ctx.energy >= ctx.max_energy:
                return {"actionType": "RECHARGE_END"}
            return work

        # Panels stowed: do blocked-while-recharging work now; otherwise only
        # deploy panels when too low on energy to mine and no threat is present.
        if not ctx.allowed_while_recharging(work_type):
            return work
        # Bank energy for a worthwhile jump: when crawling toward a distant
        # target we cannot yet afford to jump to, deploy panels so we recharge
        # WHILE we MOVE (MOVE is allowed while recharging). Once energy reaches
        # jumpMinCost the travel logic jumps the remaining distance, saving many
        # crawl ticks. Skipped near threats and once energy is already capped.
        if (
            work_type == "MOVE"
            and self.bank_for_jump
            and ctx.has_jump
            and ctx.energy < ctx.jump_min_cost
            and ctx.energy < ctx.max_energy
            and not self.threats
        ):
            logger.debug(
                "BANK energy for jump: energy=%d < jumpMinCost=%d while crawling "
                "to a distant target -> RECHARGE (jump once charged)",
                ctx.energy, ctx.jump_min_cost,
            )
            return {"actionType": "RECHARGE"}
        if ctx.energy < ctx.mine_cost and ctx.energy < ctx.max_energy and not self.threats:
            return {"actionType": "RECHARGE"}
        return work

    # --- Anti-oscillation + bounds sanitisers ---
    def _undoes_previous_step(self, action):
        ctx = self.ctx
        if ctx.prev_cell is None or action.get("actionType") != "MOVE":
            return False
        direction = action.get("payload", {}).get("direction")
        if direction not in ("N", "S", "E", "W"):
            return False
        return ShipUtils.next_position(ctx.location, direction) == ctx.prev_cell

    def _sanitize(self, action):
        ctx = self.ctx
        action_type = action.get("actionType")
        if action_type == "MOVE":
            direction = action.get("payload", {}).get("direction")
            nx, ny = ShipUtils.next_position(ctx.location, direction)
            if not ctx.in_bounds(nx, ny):
                return {"actionType": "WAIT"}
        elif action_type == "JUMP":
            tgt = action.get("payload", {}).get("target_location", {}) or {}
            tx = min(max(0, ShipUtils.safe_int(tgt.get("x"), ctx.location[0])), ctx.map_width - 1)
            ty = min(max(0, ShipUtils.safe_int(tgt.get("y"), ctx.location[1])), ctx.map_height - 1)
            if (tx, ty) == ctx.location:
                return {"actionType": "WAIT"}
            return {"actionType": "JUMP", "payload": {"target_location": {"x": tx, "y": ty}}}
        return action

    def _log_decision_inputs(self):
        """Summarise the action-request signals that drive the next choice.

        Two compact lines complement the TICK header with the inputs the policy
        actually reasons over: the mining picture (current-cell odds + the
        best-scored targets and nearest post) and the combat picture (our power
        vs. acting/dominant threats). They let post-game log analysis explain
        *why* a MINE failed, which rock was chosen and whether a fight was
        (correctly) avoided.
        """
        if not _LOGGING_ENABLED:
            return
        ctx = self.ctx

        def rock_odds(asteroid):
            nutr = ShipUtils.safe_int(asteroid.get("nutrinium"))
            mass = max(ShipUtils.safe_int(asteroid.get("mass"), 1), 1)
            density = nutr / mass
            success = min(1.0, density * 10.0 * ctx.mine_base_success)
            return nutr, mass, density, success

        # Current-cell mining odds (explains MINE success/FAIL on this rock).
        on_rock = next(
            (a for a in ctx.asteroids
             if ShipUtils.location(a) == ctx.location
             and ShipUtils.safe_int(a.get("nutrinium")) > 0),
            None,
        )
        if on_rock is not None:
            nutr, mass, density, success = rock_odds(on_rock)
            rock_str = "onRock n=%d m=%d dens=%.3f mineSucc=%.0f%%" % (
                nutr, mass, density, success * 100,
            )
        else:
            rock_str = "onRock=none"

        # Top mineable candidates by score (shows target-selection quality).
        ranked = sorted(ctx.mineable_asteroids, key=lambda a: -ctx.asteroid_score(a))[:3]
        candidates = []
        for asteroid in ranked:
            loc = ShipUtils.location(asteroid)
            nutr, mass, density, success = rock_odds(asteroid)
            candidates.append(
                "%s n=%d dens=%.2f succ=%.0f%% score=%.2f dist=%.1f" % (
                    loc, nutr, density, success * 100,
                    ctx.asteroid_score(asteroid),
                    ShipUtils.distance(ctx.location, loc),
                )
            )

        # Nearest post (drives banking / haul economics).
        nearest_post = ShipUtils.nearest(ctx.location, ctx.trading_posts)
        if nearest_post is not None:
            post_loc = ShipUtils.location(nearest_post)
            post_dist = ShipUtils.distance(ctx.location, post_loc)
            post_str = "nearestPost=%s dist=%.1f jumpCost=%d" % (
                post_loc, post_dist, ctx.jump_energy(post_dist),
            )
        else:
            post_str = "nearestPost=none"

        logger.debug(
            "SCAN %s | mineable=%d top=[%s] | %s",
            rock_str, len(ctx.mineable_asteroids), " ; ".join(candidates), post_str,
        )

        # Combat picture: our power vs. the acting/dominant threats we weigh.
        def power_list(ships):
            return ", ".join(
                "%s p=%.1f" % (ShipUtils.location(e), ctx.combat.rough_power(e))
                for e in sorted(ships, key=lambda e: -ctx.combat.rough_power(e))[:3]
            )

        logger.debug(
            "COMBAT myPower=%.1f eff=%.1f overwhelmed=%s | acting=%d sameZone=%d | "
            "threats=%d [%s] | dominant=%d [%s]",
            self.my_power, self.my_power_eff, self.is_overwhelmed,
            len(self.acting_enemies), len(ctx.same_zone_enemies),
            len(self.threats), power_list(self.threats),
            len(self.dominant_threats), power_list(self.dominant_threats),
        )

    def decide(self):
        ctx = self.ctx
        self._log_decision_inputs()
        chosen = self._choose()
        logger.debug(
            "DECIDE loc=%s prev_cell=%s energy=%s nutrinium=%s recharging=%s "
            "overwhelmed=%s threats=%d bankJump=%s chosen=%s payload=%s",
            ctx.location, ctx.prev_cell, ctx.energy, ctx.nutrinium, ctx.recharging,
            self.is_overwhelmed, len(self.threats), self.bank_for_jump,
            chosen.get("actionType"), chosen.get("payload"),
        )

        chosen = _enforce(self.ctx, chosen)

        # Damp two-cell oscillation (exempt genuine flight).
        if not self.is_overwhelmed and self._undoes_previous_step(chosen):
            logger.debug(
                "OSCILLATION damp: MOVE %s from %s would step back onto prev_cell "
                "%s -> WAIT instead",
                chosen.get("payload", {}).get("direction"),
                ctx.location, ctx.prev_cell,
            )
            chosen = {"actionType": "WAIT"}

        final = self._sanitize(chosen)
        logger.debug("DECIDE final action=%s payload=%s", final.get("actionType"), final.get("payload"))
        return final


# ---------------------------------
# Action-mask safety net (shared utils.action_masker)
# ---------------------------------
_MASKER = "unset"  # sentinel -> the utils.action_masker module, or None on failure


def _get_masker():
    """Lazily import and cache ``utils.action_masker``. Returns the module or None.

    The ``src`` directory (parent of ``bots``) is added to ``sys.path`` so the
    utility resolves both inside the simulator and when run as a standalone
    lambda. Any failure (e.g. numpy missing) degrades gracefully -- the caller
    keeps the heuristic's action unmasked.
    """
    global _MASKER
    if _MASKER != "unset":
        return _MASKER
    result = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(here)  # .../src
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from utils import action_masker
        result = action_masker
    except Exception:
        result = None
    _MASKER = result
    return result


def _flatten_entities(contacts, keys):
    """Flatten raw sensor contacts (nested ``location``) to ``{'x','y',...}`` dicts."""
    out = []
    for contact in contacts:
        loc = contact.get("location", {}) or {}
        entity = {"x": ShipUtils.safe_int(loc.get("x")), "y": ShipUtils.safe_int(loc.get("y"))}
        for key in keys:
            entity[key] = contact.get(key)
        out.append(entity)
    return out


def _build_mask_state(ctx, masker):
    """Adapt the parsed :class:`GameContext` into a ``MaskState``."""
    energy_costs = {
        "mine": ctx.mine_cost,
        "move": ctx.move_cost,
        "attack": ctx.attack_cost,
        "shields": ctx.shields_cost,
        "jump": int(ctx.jump_cost_per_unit),
        "plunder": ctx.plunder_cost,
        "negotiate": ctx.negotiate_cost,
    }
    negotiate_post_id = ctx.objective_post.get("id") if ctx.objective_post else None
    return masker.MaskState(
        x=ctx.location[0],
        y=ctx.location[1],
        energy=ctx.energy,
        health=ctx.health,
        nutrinium=ctx.nutrinium,
        credits=ctx.credits,
        destroyed=(ctx.state == "DESTROYED"),
        recharging=ctx.recharging,
        just_recharged=False,
        shield_state=ctx.shield_state,
        shield_value=ctx.shield_value,
        shield_capacity=ctx.shield_capacity,
        shields_up=ctx.shields_up,
        modules=ctx.modules,
        negotiate_post_id=negotiate_post_id,
        enemies=_flatten_entities(ctx.enemy_ships, ("nutrinium", "playerId", "shield")),
        asteroids=_flatten_entities(ctx.asteroids, ("nutrinium", "mass")),
        trading_posts=_flatten_entities(ctx.trading_posts, ("id", "name")),
        wreckage=_flatten_entities(ctx.wreckage, ("nutrinium",)),
        map_width=ctx.map_width,
        map_height=ctx.map_height,
        max_energy=ctx.max_energy,
        max_health=ctx.max_health,
        energy_costs=energy_costs,
        salvage_energy_cost=ctx.salvage_energy_cost,
        repair_cost=0,
        action_restrictions=ctx.action_restrictions,
    )


def _action_name_to_id(masker, action, ctx):
    """Map a response dict back to the env action id the masker validates."""
    action_type = action.get("actionType")
    payload = action.get("payload", {}) or {}
    simple = {
        "WAIT": masker.WAIT,
        "MINE": masker.MINE,
        "RECHARGE": masker.RECHARGE,
        "RECHARGE_END": masker.RECHARGE_END,
        "SELL": masker.SELL,
        "ATTACK": masker.ATTACK,
        "PLUNDER": masker.PLUNDER,
        "RESPAWN": masker.RESPAWN,
        "RAISE_SHIELDS": masker.RAISE_SHIELDS,
        "LOWER_SHIELDS": masker.LOWER_SHIELDS,
        "NEGOTIATE": masker.NEGOTIATE,
        "SALVAGE": masker.SALVAGE,
        "REPAIR": masker.REPAIR,
    }
    if action_type in simple:
        return simple[action_type]
    if action_type == "MOVE":
        return {
            "N": masker.MOVE_NORTH,
            "S": masker.MOVE_SOUTH,
            "E": masker.MOVE_EAST,
            "W": masker.MOVE_WEST,
        }.get(payload.get("direction"), masker.WAIT)
    if action_type == "JUMP":
        tgt = payload.get("target_location", {}) or {}
        tx = ShipUtils.safe_int(tgt.get("x"), ctx.location[0])
        ty = ShipUtils.safe_int(tgt.get("y"), ctx.location[1])
        if any(ShipUtils.location(p) == (tx, ty) for p in ctx.trading_posts):
            return masker.JUMP_TO_TRADING_POST
        return masker.JUMP_TO_ASTEROID
    return masker.WAIT


def _masked_to_response(ctx, action_id, masker):
    """Rebuild a response dict for an enforced (valid) action id."""
    moves = {
        masker.MOVE_NORTH: "N",
        masker.MOVE_SOUTH: "S",
        masker.MOVE_EAST: "E",
        masker.MOVE_WEST: "W",
    }
    if action_id in moves:
        return {"actionType": "MOVE", "payload": {"direction": moves[action_id]}}
    if action_id == masker.MINE:
        return {"actionType": "MINE"}
    if action_id == masker.RECHARGE:
        return {"actionType": "RECHARGE"}
    if action_id == masker.RECHARGE_END:
        return {"actionType": "RECHARGE_END"}
    if action_id == masker.RESPAWN:
        return {"actionType": "RESPAWN"}
    if action_id == masker.RAISE_SHIELDS:
        return {"actionType": "RAISE_SHIELDS"}
    if action_id == masker.SELL:
        return {"actionType": "SELL", "payload": {"nutrinium": ctx.nutrinium}}
    if action_id == masker.ATTACK:
        enemies = ctx.same_zone_enemies
        if not enemies:
            return {"actionType": "WAIT"}
        richest = max(enemies, key=lambda e: ShipUtils.safe_int(e.get("nutrinium")))
        target = richest.get("playerId")
        if target is None:
            return {"actionType": "WAIT"}
        return {"actionType": "ATTACK",
                "payload": {"target": target, "energy": min(20, ctx.energy)}}
    if action_id == masker.JUMP_TO_ASTEROID:
        nearest = ShipUtils.nearest(ctx.location, ctx.mineable_asteroids)
        if nearest is None:
            return {"actionType": "WAIT"}
        tx, ty = ShipUtils.location(nearest)
        return {"actionType": "JUMP", "payload": {"target_location": {"x": tx, "y": ty}}}
    if action_id == masker.JUMP_TO_TRADING_POST:
        nearest = ShipUtils.nearest(ctx.location, ctx.trading_posts)
        if nearest is None:
            return {"actionType": "WAIT"}
        tx, ty = ShipUtils.location(nearest)
        return {"actionType": "JUMP", "payload": {"target_location": {"x": tx, "y": ty}}}
    return {"actionType": "WAIT"}


def _enforce(ctx, action):
    """Validate the heuristic's action via the shared masker; substitute if invalid.

    When the masking utility cannot be imported, the heuristic's action is kept
    unchanged so the bot still functions as a pure-stdlib lambda.
    """
    masker = _get_masker()
    if masker is None:
        return action
    state = _build_mask_state(ctx, masker)
    action_id = _action_name_to_id(masker, action, ctx)
    is_valid, _ = masker.is_action_valid(action_id, state)
    if is_valid:
        return action
    enforced_id = masker.mask_action(action_id, state)
    enforced = _masked_to_response(ctx, enforced_id, masker)
    logger.warning(
        "Heuristic chose invalid action %r; action mask enforced %r.",
        action.get("actionType"), enforced.get("actionType"),
    )
    return enforced


# --------------------------------------------------------------------------
# Public lambda contract
# --------------------------------------------------------------------------
def get_action(action_request):
    logger.info(action_request)
    action = get_heuristic_action(action_request)
    return _to_response(action)


def _log_round_state(ctx):
    """Once per game+round, log the full round setup the bot will play under.

    A single ROUND-START line captures the fixed parameters (map, costs, ship
    caps, mining/market economics, objective) plus our opening ship state and
    so a finished round's conditions can be reviewed without inferring them
    from per-tick traces.
    """
    if not _LOGGING_ENABLED or ctx.game_id is None:
        return
    key = (ctx.game_id, ctx.round)
    if key in _round_start_logged:
        return
    _round_start_logged.add(key)

    md = ctx.metadata
    ship_cfg = md.get("shipConfig", {}) or {}
    map_cfg = md.get("mapConfig", {}) or {}
    market = md.get("market", {}) or {}
    sell = market.get("sell", {}) or {}
    buy = market.get("buy", {}) or {}
    sensor_range = (ship_cfg.get("sensors", {}) or {}).get("range")
    energy_per_recharge = ship_cfg.get("energyPerRecharge")
    sell_cost = (ship_cfg.get("energyCosts", {}) or {}).get("sell", 0)
    entry = ctx.my_leaderboard_entry() or {}
    ticks_left = "n/a" if ctx.ticks_remaining is None else "%.1f" % ctx.ticks_remaining

    logger.debug(
        "ROUND-START game=%s round=%s tick=%s | map=%sx%s astDensity=%s posts=%s "
        "sensorRange=%s | costs(mine=%s move=%s jump/u=%s jumpMin=%s plunder=%s "
        "negotiate=%s attack=%s sell=%s) | ship(maxEnergy=%s maxJump=%s "
        "perRecharge=%s) | mining(baseSucc=%s payout=%s-%s mod=%s) | "
        "market(sellNutr=%s repair=%s ship=%s) | "
        "me(loc=%s energy=%s health=%s nutrinium=%s credits=%s shields=%s "
        "modules=%s skillPts=%s/%s) | objective=%r | "
        "field(enemies=%s asteroidsMineable=%s/%s posts=%s wreckage=%s) | "
        "leaderboard(size=%s myPos=%s myScore=%s) | ticks_left=%s",
        ctx.game_id, ctx.round, ctx.tick,
        ctx.map_width, ctx.map_height, map_cfg.get("asteroidDensity"),
        map_cfg.get("tradingPostCount"), sensor_range,
        ctx.mine_cost, ctx.move_cost, ctx.jump_cost_per_unit, ctx.jump_min_cost,
        ctx.plunder_cost, ctx.negotiate_cost, ctx.attack_cost, sell_cost,
        ctx.max_energy, ctx.max_jump_distance, energy_per_recharge,
        ctx.mine_base_success, ctx.mine_min_payout, ctx.mine_max_payout,
        (md.get("mining", {}) or {}).get("payoutModifier"),
        sell.get("nutrinium"), buy.get("repair"), buy.get("ship"),
        ctx.location, ctx.energy, ctx.health, ctx.nutrinium, ctx.credits,
        "UP" if ctx.shields_up else "DOWN", ctx.modules,
        ctx.skill_points_spent, ctx.skill_points_total,
        ctx.objective_post_name,
        len(ctx.enemy_ships), len(ctx.mineable_asteroids), len(ctx.asteroids),
        len(ctx.trading_posts), len(ctx.wreckage),
        len(ctx.leaderboard) or "?", entry.get("position", "?"),
        entry.get("gameScore", "?"), ticks_left,
    )


def _log_endgame_stats(ctx):
    """Once per game+round, when the round clock is about to expire, log final stats.

    Captures the ship's outcome (credits, leaderboard rank, score) and the
    cumulative combat/mining/salvage/repair tallies so a finished round can be
    reviewed without replaying every tick.
    """
    if not _LOGGING_ENABLED or ctx.game_id is None:
        return
    key = (ctx.game_id, ctx.round)
    if key in _endgame_logged:
        return
    if ctx.ticks_remaining is None or ctx.ticks_remaining > 1:
        return
    _endgame_logged.add(key)

    entry = ctx.my_leaderboard_entry() or {}
    combat = ctx.stats.get("combat", {}) or {}
    mining = ctx.stats.get("mining", {}) or {}
    salvage = ctx.stats.get("salvage", {}) or {}
    repair = ctx.stats.get("repair", {}) or {}
    logger.debug(
        "ENDGAME game=%s round=%s player=%s ship=%r | rank=%s/%s gameScore=%s credits=%s "
        "nutrinium=%s health=%s skillPts=%s/%s | "
        "finalLoc=%s energy=%s recharging=%s shields=%s | "
        "field(enemies=%s asteroidsMineable=%s/%s posts=%s) | "
        "mined=%s mineSuccess=%s/%s | dmgDealt=%s dmgTaken=%s dmgBlocked=%s "
        "kills=%s respawns=%s nutrWon=%s nutrLost=%s | "
        "salvaged=%s repairHealth=%s | roundScores=%s",
        ctx.game_id, ctx.round, ctx.player_id, ctx.me.get("name"),
        entry.get("position", "?"), len(ctx.leaderboard) or "?",
        entry.get("gameScore", "?"), ctx.credits, ctx.nutrinium, ctx.health,
        ctx.skill_points_spent, ctx.skill_points_total,
        ctx.location, ctx.energy, ctx.recharging,
        "UP" if ctx.shields_up else "DOWN",
        len(ctx.enemy_ships), len(ctx.mineable_asteroids), len(ctx.asteroids),
        len(ctx.trading_posts),
        mining.get("nutriniumMined"), mining.get("success"), mining.get("attempts"),
        combat.get("damageDealt"), combat.get("damageTaken"), combat.get("damageBlocked"),
        combat.get("destroyed"), combat.get("respawns"),
        combat.get("nutriniumWon"), combat.get("nutriniumLost"),
        salvage.get("nutriniumSalvaged"), repair.get("healthRestored"),
        ctx.round_scores,
    )


def get_heuristic_action(action_request):
    ctx = GameContext(action_request)
    _ensure_game_log_file(ctx.game_id, ctx.round)
    _log_round_state(ctx)

    # Learn the real N/S/E/W orientation from our last move and use our own
    # remembered previous cell (more reliable than reconstructing it from a
    # payload-less actionResult) so anti-oscillation can fire.
    own_prev = _calibrate_from_last_move(ctx)
    if own_prev is not None:
        ctx.prev_cell = None if own_prev == ctx.location else own_prev
        ctx.prev_cell_source = "own-move"
    logger.debug(
        "PREV_CELL %s (source=%s)%s",
        ctx.prev_cell, ctx.prev_cell_source,
        "" if ctx.prev_cell is not None
        else " -- anti-oscillation cannot fire without a known previous cell",
    )

    # Track no-progress ticks so the strategy can break a server-rejection
    # deadlock (sets ctx.stuck_count).
    _update_stuck_state(ctx)

    ticks_left = "n/a" if ctx.ticks_remaining is None else "%.1f" % ctx.ticks_remaining
    logger.debug(
        "TICK game=%s tick=%s round=%s loc=%s state=%s energy=%s nutrinium=%s "
        "health=%s recharging=%s shields=%s modules=%s axis=ns%+d/ew%+d | "
        "enemies=%d sameZone=%d asteroids=%d posts=%d ticks_left=%s | "
        "last=%s/%s reason=%s/%s stuck=%d offenceLost=%s",
        ctx.game_id, ctx.tick, ctx.round, ctx.location, ctx.state, ctx.energy,
        ctx.nutrinium, ctx.health, ctx.recharging,
        "UP" if ctx.shields_up else "DOWN", ctx.modules,
        _axis["ns"], _axis["ew"],
        len(ctx.enemy_ships), len(ctx.same_zone_enemies),
        len(ctx.mineable_asteroids), len(ctx.trading_posts), ticks_left,
        ctx.last_action_type or "-", "FAIL" if ctx.last_action_failed else "ok",
        ctx.last_action_result_code or "-", ctx.last_action_message or "-",
        ctx.stuck_count, ctx.offence_target_lost,
    )

    action = HeuristicStrategy(ctx).decide()
    _record_move(ctx, action)
    _log_endgame_stats(ctx)
    return action