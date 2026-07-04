"""Prospectors & Pirates bot (v5).

A standalone strategy port of the 2025 ``r680329`` competition bot
(``spaceship.SpaceShip`` + ``action.Action``) onto the 2026 lambda contract
shared by ``bot_v2`` / ``bot_v3`` / ``bot_v4``: a single :func:`get_action` that
takes an ActionRequest dict and returns ``{"actionType": str, "payload"?: ...}``.

The 2025 bot is a BALANCED MINER-TRADER with opportunistic raiding and -- its
signature -- market-timing on sells (bank cargo when the price is high relative
to the round's running low/high). This port keeps that personality while
adapting to the 2026 game model:

* The 2025 action-point budget is gone (2026 is one action per tick) -- all
  action-point bookkeeping is dropped.
* The 2025 COMBAT sub-system (allocating energy to WEAPONS/SHIELDS/THRUSTERS) is
  collapsed onto the 2026 single-shot ``ATTACK{target, energy}``, sizing the
  energy from the same energy-matchup prey-selection rules.
* The 2025 dense ``long`` sensor matrix is replaced by the 2026 flat ``sensors``
  list (entities within sensor range); the zone-valuation formula is applied per
  visible asteroid.
* 2026-only mobility (``JUMP``) is used to reach far targets, like bot_v3/v4.

Decision priority (mirrors ``generate_action_response``):

1. Respawn if destroyed.
2. Energy management: recharge when low, end recharge once topped up.
3. Threat-aware banking: carrying cargo + a stronger ship nearby -> sell on a
   post, else flee toward the nearest post.
4. Market-timing sell: when the price/round-history says it's time, sell on a
   post (navigate to one otherwise).
5. Opportunistic attack: overpower a profitable, beatable ship in our cell.
6. Mine-vs-move: mine the asteroid under us when it's our target zone or rich
   enough, otherwise travel toward the best zone (jump when far + affordable).

Directions are emitted in the live server frame (N=y+1, S=y-1, E=x+1, W=x-1),
matching the environment's MOVE actions, so no axis calibration is needed.

NOTE: like the other bots, this one only sees what the server reports in
``sensors`` (entities within sensor range), so its choices are necessarily local
compared to the in-environment AIs that have global map knowledge.
"""

import os
import random
import sys


def _to_response(payload):
    """Lightweight lambda response adapter (mirrors bot_v2/bot_v3/bot_v4)."""
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


# ----------------------------------------------------------------------------
# Tunables (ported from the 2025 bot, adapted to the 2026 energy economy)
# ----------------------------------------------------------------------------
RECHARGE_LOW_FRAC = 0.10      # recharge when energy drops below 10% of max
RECHARGE_END_FRAC = 0.60      # stop recharging once back to ~60% of max
NUTRINIUM_MIN_RATIO = 0.7     # paid-mine in place when concentration (nutr/mass) > 0.7
# Selling dips the shared market price by a fixed fraction PER SALE (independent
# of quantity) and it recovers only slowly, so many small sales crater the price
# while a few large batches keep it near base. We therefore hold cargo into big
# batches. (A post is essentially always in sensor range, so banking stays
# reliable even with a high threshold.)
MIN_SELL_QUANTITY = int(os.environ.get("V5_OPP", "30"))    # min cargo before an on-post sale
BANK_CARGO_QUANTITY = int(os.environ.get("V5_BANK", "40")) # commit to a banking run at/above this
HARD_SELL_QUANTITY = 60       # drop everything and bank once cargo reaches this
HAUL_CARGO_THRESHOLD = 12     # cargo worth carrying toward a post when idle
MAX_MINING_SEARCH_TICK = 10   # ignore candidate zones beyond this Manhattan range
DEFAULT_ATTACK_ENERGY = 20    # baseline ATTACK payload
ATTACK_CRITICAL_MARGIN = 5    # extra energy committed to ensure an overpower

# --- Plunder evasion --------------------------------------------------------
# PLUNDER steals cargo from a ship whose shields are DOWN when an enemy shares
# its exact cell -- it does NOT depend on relative energy, so even a weak raider
# can rob us. Defences (cheapest first): sell/empty the cargo, raise shields to
# become un-plunderable, or refuse to co-locate with an enemy while loaded.
PLUNDER_PROTECT_MIN = 10      # only bother defending cargo once it's worth this much
SHIELD_PROXIMITY_RADIUS = 1   # Chebyshev range at which we proactively shield up
NEARBY_ENEMY_RADIUS = 3       # an enemy within this range makes us bank cargo sooner
NEARBY_ENEMY_BANK_FRAC = 0.5  # scale the bank threshold by this when a raider is near

# --- Free-mining regime -----------------------------------------------------
# The engine credits the per-tick recharge gain BEFORE the action resolves and
# permits MINE while recharging, so a mine taken during recharge is effectively
# free (the +energyPerRecharge gain pays the mine cost). Staying in the
# recharge state while mining therefore roughly doubles mining throughput vs.
# the classic "mine down, then sit idle recharging" cycle. The masker only lets
# us START recharging at <=30% energy, so we spend headroom first, then enter
# the regime and mine for free from then on.
RECHARGE_ENTER_FRAC = 0.30    # start recharging at/below 30% of max (masker cap)
FREE_MINE_MIN_RATIO = float(os.environ.get("V5_RATIO", "0.35"))  # keep free-mining while success >= this


# ----------------------------------------------------------------------------
# Optional id hooks (extensible; empty by default). The 2025 bot kept hardcoded
# "friend"/"attacker" player-id lists; they are irrelevant to the 2026 simulator
# and lambda, so threat/prey decisions here are purely energy-based.
# ----------------------------------------------------------------------------
_FRIEND_IDS = set()
_ATTACKER_IDS = set()


# ----------------------------------------------------------------------------
# Module-level market history (the 2025 buy-low / sell-high memory).
# Keyed on gameId so interleaved opponents in the simulator don't clobber each
# other; reset when the round changes (matches the 2025 per-round reset).
# ----------------------------------------------------------------------------
_market_high = {}   # gameId -> highest sell price seen this round
_market_low = {}    # gameId -> lowest sell price seen this round
_market_round = {}  # gameId -> round the history above belongs to


def _reset_market_state():
    """Clear the market-history globals (test isolation helper)."""
    _market_high.clear()
    _market_low.clear()
    _market_round.clear()


def _update_market_history(game_id, game_round, price):
    """Track per-round running high/low of the market sell price."""
    if price is None or price <= 0:
        return
    if _market_round.get(game_id) != game_round:
        _market_round[game_id] = game_round
        _market_high[game_id] = price
        _market_low[game_id] = price
        return
    if price > _market_high.get(game_id, price):
        _market_high[game_id] = price
    if price < _market_low.get(game_id, price):
        _market_low[game_id] = price


# ----------------------------------------------------------------------------
# Geometry / parsing helpers (stateless)
# ----------------------------------------------------------------------------
def _distance(x1, y1, x2, y2):
    """Euclidean distance between two cells."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _manhattan(x1, y1, x2, y2):
    """Manhattan distance between two cells (zone-valuation tick proxy)."""
    return abs(x1 - x2) + abs(y1 - y2)


def _skill(skills, name):
    """Skill/ability level (0 when absent)."""
    try:
        return int(skills.get(name, 0) or 0)
    except (AttributeError, TypeError, ValueError):
        return 0


def _entity_at(x, y, entities):
    """First entity exactly at (x, y), or None."""
    for e in entities:
        if e["x"] == x and e["y"] == y:
            return e
    return None


def _entities_at(x, y, entities):
    """All entities exactly at (x, y)."""
    return [e for e in entities if e["x"] == x and e["y"] == y]


def _nearest_entity(x, y, entities):
    """Nearest entity to (x, y) by Euclidean distance, or None."""
    best = None
    best_dist = None
    for e in entities:
        d = _distance(x, y, e["x"], e["y"])
        if best_dist is None or d < best_dist:
            best_dist = d
            best = e
    return best


def _direction_towards(dx, dy):
    """Compass direction to step along the dominant axis (server frame)."""
    if abs(dx) > abs(dy):
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


def _zone_value(nutrinium, mass, ship_count, dist):
    """2025 negative-compounding-interest zone valuation.

    Rewards rich, high-concentration asteroids that are close and uncontested.
    Distance and competitors decay the value sharply (the exponent grows with
    both), so a near, rich, uncontested rock dominates a far/poor/crowded one.
    """
    if mass <= 0 or nutrinium <= 0:
        return 0.0
    ratio = nutrinium / mass
    tick_count = 1 + dist
    decay_factor = (2 * tick_count) + ship_count
    if decay_factor <= 0:
        return 0.0
    base = 1.0 - (ratio / decay_factor)
    if base < 0.0:
        base = 0.0
    try:
        return nutrinium * (base ** (decay_factor * decay_factor)) * ratio
    except (OverflowError, ValueError):
        return 0.0


# ----------------------------------------------------------------------------
# Parsed request
# ----------------------------------------------------------------------------
class _Context:
    """Parses one ActionRequest into the fields the balanced loop needs."""

    def __init__(self, action_request):
        req = action_request or {}
        me = req.get("me", {}) or {}
        loc = me.get("location", {}) or {}
        self.x = int(loc.get("x", 0))
        self.y = int(loc.get("y", 0))
        self.energy = int(me.get("energy", 0))
        self.health = int(me.get("health", 100))
        self.nutrinium = int(me.get("nutrinium", 0))
        self.recharging = bool(me.get("recharging", False))
        self.state = str(me.get("state", "READY")).upper()
        self.skills = me.get("skills", {}) or {}
        # Installed modules (e.g. JUMP) gate module-locked actions in the masker.
        self.modules = {str(m).upper() for m in (me.get("modules", []) or [])}
        self.player_id = me.get("playerId")

        # Own shield: shields POWERED/DRAINING make us immune to PLUNDER.
        shield = me.get("shield", {}) or {}
        self.shield_state = str(shield.get("state", "DOWN") or "DOWN").upper()
        self.shield_value = float(shield.get("value", 0) or 0)
        self.shield_capacity = float(shield.get("capacity", 0) or 0)
        self.shields_up = self.shield_state == "POWERED"

        # Split sensor contacts by type into flat {x, y, ...} dicts.
        self.asteroids = []
        self.trading_posts = []
        self.ships = []
        for s in req.get("sensors", []) or []:
            s_loc = s.get("location", {}) or {}
            entry = {"x": int(s_loc.get("x", 0)), "y": int(s_loc.get("y", 0))}
            s_type = s.get("type")
            if s_type == "asteroid":
                entry["nutrinium"] = int(s.get("nutrinium", 0))
                entry["mass"] = int(s.get("mass", 0))
                self.asteroids.append(entry)
            elif s_type == "trading_post":
                entry["name"] = s.get("name")
                entry["id"] = s.get("id")
                self.trading_posts.append(entry)
            elif s_type == "ship":
                if str(s.get("state", "")).upper() == "DESTROYED":
                    continue
                entry["health"] = int(s.get("health", 100))
                entry["energy"] = int(s.get("energy", 0))
                entry["nutrinium"] = int(s.get("nutrinium", 0))
                entry["playerId"] = s.get("playerId")
                s_shield = s.get("shield", {}) or {}
                entry["shield_state"] = str(s_shield.get("state", "") or "").upper()
                self.ships.append(entry)

        # Total visible mineable nutrinium -> MINING vs ATTACKING mode proxy.
        self.total_remaining = sum(a["nutrinium"] for a in self.asteroids)

        # Round metadata: economics + map size + market price.
        game_state = req.get("gameState", {}) or {}
        self.game_id = game_state.get("gameId")
        self.game_round = game_state.get("round", 0)
        self.tick = int(game_state.get("tick", 0) or 0)
        metadata = game_state.get("metadata", {}) or {}
        ship_cfg = metadata.get("shipConfig", {}) or {}
        costs = ship_cfg.get("energyCosts", {}) or {}
        self.mine_cost = int(costs.get("mine", 10))
        self.attack_cost = int(costs.get("attack", 1))
        self.jump_unit_cost = int(costs.get("jump", 1))
        self.jump_min_cost = int(costs.get("jumpMinCost", 75))
        self.max_energy = int(ship_cfg.get("maxEnergy", 100))
        self.max_jump_distance = int(ship_cfg.get("maxJumpDistance", 50))
        self.per_recharge = int(ship_cfg.get("energyPerRecharge", 10))
        # Extra fields the shared action-masker safety net validates against.
        self.move_cost = int(costs.get("move", 1))
        self.shields_cost = int(costs.get("shields", 5))
        self.plunder_cost = int(costs.get("plunder", 5))
        self.negotiate_cost = int(costs.get("negotiate", 5))
        self.max_health = int(ship_cfg.get("maxHealth", 100))
        self.credits = int(me.get("credits", 0))
        map_cfg = metadata.get("mapConfig", {}) or {}
        self.map_w = int(map_cfg.get("width", 10))
        self.map_h = int(map_cfg.get("height", 10))
        market = metadata.get("market", {}) or {}
        sell_cfg = market.get("sell", {}) or {}
        price = sell_cfg.get("nutrinium")
        self.market_price = float(price) if price is not None else 0.0
        # Per-state action restrictions (allowedWhileRecharging / allowedWithShieldsUp).
        self.action_restrictions = metadata.get("actionRestrictions", {}) or {}

    def jump_energy_cost(self, distance):
        """Energy cost of a jump of the given distance (skill lowers the floor)."""
        adj_min_cost = max(0, self.jump_min_cost - _skill(self.skills, "jump_cost") * 5)
        return int(max(adj_min_cost, round(self.jump_unit_cost * distance)))


# ----------------------------------------------------------------------------
# Action builders
# ----------------------------------------------------------------------------
def _move(direction):
    return {"actionType": "MOVE", "payload": {"direction": direction}}


def _jump(target):
    return {"actionType": "JUMP",
            "payload": {"target_location": {"x": target["x"], "y": target["y"]}}}


def _sell(amount):
    return {"actionType": "SELL", "payload": {"nutrinium": int(amount)}}


def _attack(target, energy):
    return {"actionType": "ATTACK",
            "payload": {"target": target.get("playerId"), "energy": int(energy)}}


def _raise_shields():
    return {"actionType": "RAISE_SHIELDS"}


def _lower_shields():
    return {"actionType": "LOWER_SHIELDS"}


def _bank_sell(ctx):
    """Sell cargo, or end recharge first if the solar panels are deployed.

    SELL is masked while recharging (allowedWhileRecharging=False) and the game
    server rejects it outright, so we must drop out of the recharge state before
    selling. RECHARGE_END is legal while recharging; the sale then lands next
    tick. Not relying on the shared masker to do this keeps banking correct even
    when the masker is unavailable in production.
    """
    if ctx.recharging:
        return {"actionType": "RECHARGE_END"}
    return _sell(ctx.nutrinium)


def _navigate(ctx, tx, ty, jump_min_dist, jump_margin):
    """Travel toward (tx, ty): jump when far and affordable, else step.

    JUMP is disallowed while recharging (solar panels deployed ->
    actionRestrictions JUMP.allowedWhileRecharging == False); the server rejects
    such a jump. So while recharging we always crawl with MOVE, which restores
    energy en route and stays inside the recharge regime.
    """
    dist = _distance(ctx.x, ctx.y, tx, ty)
    if (not ctx.recharging
            and dist > jump_min_dist
            and ctx.energy >= ctx.jump_energy_cost(dist) + jump_margin):
        return _jump({"x": tx, "y": ty})
    return _move(_direction_towards(tx - ctx.x, ty - ctx.y))


def _move_explore(ctx):
    """Step in a random in-bounds direction (2025 ``__move_random``)."""
    options = []
    if ctx.y < ctx.map_h - 1:
        options.append("N")
    if ctx.y > 0:
        options.append("S")
    if ctx.x < ctx.map_w - 1:
        options.append("E")
    if ctx.x > 0:
        options.append("W")
    if not options:
        return {"actionType": "WAIT"}
    return _move(random.choice(options))


# ----------------------------------------------------------------------------
# Strategy sub-procedures
# ----------------------------------------------------------------------------
def _has_potential_attacker(ctx):
    """True when a visible ship clearly outguns us (2025 ``__has_potential_attacker``)."""
    if any(s.get("playerId") in _ATTACKER_IDS for s in ctx.ships):
        return True
    strongest = None
    for s in ctx.ships:
        if strongest is None or s["energy"] > strongest["energy"]:
            strongest = s
    return strongest is not None and ctx.energy * 2 < strongest["energy"]


def _find_target_zone(ctx):
    """Best mining zone among visible asteroids by 2025 zone valuation."""
    best = None
    best_value = 0.0
    for ast in ctx.asteroids:
        if ast["nutrinium"] <= 0 or ast["mass"] <= 0:
            continue
        dist = _manhattan(ctx.x, ctx.y, ast["x"], ast["y"])
        if dist > MAX_MINING_SEARCH_TICK:
            continue
        ship_count = len(_entities_at(ast["x"], ast["y"], ctx.ships))
        value = _zone_value(ast["nutrinium"], ast["mass"], ship_count, dist)
        if value > best_value:
            best_value = value
            best = {"x": ast["x"], "y": ast["y"], "value": value}
    return best


def _find_prey(ctx):
    """Pick a beatable, profitable ship in our cell; size the ATTACK energy.

    Ports the 2025 energy-matchup prey selection: prefer the richest cargo we
    can overpower (their energy <= ours). In MINING mode only interrupt mining
    for a target richer than what the asteroid under us is worth.

    Returns ``(prey_or_None, energy_to_commit)``.
    """
    if ctx.energy < ctx.attack_cost:
        return None, 0
    same_zone = _entities_at(ctx.x, ctx.y, ctx.ships)
    candidates = [
        s for s in same_zone
        if s.get("playerId") != ctx.player_id
        and s.get("playerId") not in _FRIEND_IDS
    ]
    if not candidates:
        return None, 0

    mining_mode = ctx.total_remaining > 0
    asteroid_here = _entity_at(ctx.x, ctx.y, ctx.asteroids)
    if mining_mode and asteroid_here and asteroid_here["mass"] > 0:
        worth = asteroid_here["nutrinium"] ** 2 / asteroid_here["mass"]
        candidates = [s for s in candidates if s["nutrinium"] > worth]
    else:
        candidates = [s for s in candidates if s["nutrinium"] > 0]

    # Richest cargo first, then require we can overpower them on energy.
    candidates.sort(key=lambda s: s["nutrinium"], reverse=True)
    for prey in candidates:
        if prey["energy"] <= ctx.energy:
            commit = min(ctx.energy,
                         max(DEFAULT_ATTACK_ENERGY, prey["energy"] + ATTACK_CRITICAL_MARGIN))
            return prey, commit
    return None, 0


def _is_selling_time(ctx):
    """Decide whether to make a banking run.

    Selling is free and the market barely moves in this environment, so the
    dominant cost of holding cargo is the round ending with unsold nutrinium
    (pure wasted mining). We therefore bank decisively once cargo is worth a
    trip, using the 2025 market signal only to ACCELERATE a sell, never to
    delay one below the bank threshold.
    """
    if ctx.nutrinium >= HARD_SELL_QUANTITY:
        return True
    if ctx.nutrinium >= BANK_CARGO_QUANTITY:
        return True
    if ctx.nutrinium < MIN_SELL_QUANTITY:
        return False
    # Middle band (MIN_SELL..BANK): only bank early when the price is favourable.
    price = ctx.market_price
    if price <= 0:
        # No market signal -> bank on cargo size alone.
        return ctx.nutrinium >= HARD_SELL_QUANTITY
    high = _market_high.get(ctx.game_id, price)
    low = _market_low.get(ctx.game_id, price)
    if ctx.nutrinium > MIN_SELL_QUANTITY and price >= high:
        return True
    if price >= low * 1.3:
        return True
    return False


def _go_to_post(ctx):
    """Navigate to the nearest visible trading post, or None if none visible."""
    post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
    if post is None:
        return None
    return _navigate(ctx, post["x"], post["y"], jump_min_dist=1, jump_margin=5)


# ----------------------------------------------------------------------------
# Plunder evasion
# ----------------------------------------------------------------------------
_MOVE_DELTAS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}


def _in_bounds(ctx, x, y):
    return 0 <= x < ctx.map_w and 0 <= y < ctx.map_h


def _valid_move_dirs(ctx):
    """In-bounds MOVE directions from the current cell."""
    return [d for d, (dx, dy) in _MOVE_DELTAS.items()
            if _in_bounds(ctx, ctx.x + dx, ctx.y + dy)]


def _raiders(ctx):
    """Hostile ships that can afford to PLUNDER (energy >= plunder cost)."""
    return [s for s in ctx.ships
            if s.get("playerId") != ctx.player_id
            and s.get("playerId") not in _FRIEND_IDS
            and s["energy"] >= ctx.plunder_cost]


def _plunder_threat(ctx):
    """Raiders that put our cargo at risk. Returns ``(co_located, adjacent)``.

    PLUNDER needs an enemy on our exact cell while our shields are DOWN and is
    energy-independent, so any raider with enough energy is a threat. Only
    meaningful once we carry cargo worth protecting.
    """
    if ctx.nutrinium < PLUNDER_PROTECT_MIN:
        return None, None
    co_located = None
    adjacent = None
    for s in _raiders(ctx):
        cheb = max(abs(s["x"] - ctx.x), abs(s["y"] - ctx.y))
        if cheb == 0:
            co_located = co_located or s
        elif cheb <= SHIELD_PROXIMITY_RADIUS:
            adjacent = adjacent or s
    return co_located, adjacent


def _raider_nearby(ctx):
    """True when a plunder-capable enemy is within NEARBY_ENEMY_RADIUS."""
    return any(max(abs(s["x"] - ctx.x), abs(s["y"] - ctx.y)) <= NEARBY_ENEMY_RADIUS
               for s in _raiders(ctx))


def _can_shield(ctx):
    """Whether RAISE_SHIELDS is affordable/legal this tick (makes us un-plunderable).

    RAISE_SHIELDS is masked while recharging (allowedWhileRecharging=False) and a
    zero-capacity ship can't shield, so both cases fall back to positional evasion.
    """
    return (not ctx.recharging
            and ctx.shield_capacity > 0
            and ctx.shield_state != "POWERED"
            and ctx.energy >= ctx.shields_cost)


def _flee_toward_post(ctx, avoid):
    """Step toward the nearest post without landing on any cell in ``avoid``.

    Falls back to the safe in-bounds move that most increases distance from the
    avoided cells, then WAIT if boxed in. MOVE works while recharging.
    """
    avoid = set(avoid)
    safe = [d for d in _valid_move_dirs(ctx)
            if (ctx.x + _MOVE_DELTAS[d][0], ctx.y + _MOVE_DELTAS[d][1]) not in avoid]
    if not safe:
        return {"actionType": "WAIT"}
    post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
    if post is not None:
        return _move(min(
            safe,
            key=lambda d: _distance(ctx.x + _MOVE_DELTAS[d][0],
                                    ctx.y + _MOVE_DELTAS[d][1], post["x"], post["y"]),
        ))
    # No post visible: just maximise distance from the avoided cells.
    if not avoid:
        return _move(safe[0])
    return _move(max(
        safe,
        key=lambda d: min(_distance(ctx.x + _MOVE_DELTAS[d][0],
                                    ctx.y + _MOVE_DELTAS[d][1], ax, ay) for ax, ay in avoid),
    ))


def _evade_plunder(ctx, on_post):
    """Protect cargo from raiders. Returns an action, or None if no threat.

    Priority: bank the cargo (selling empties the prize), else shield up to
    become un-plunderable, else move off/away from the raider's cell.
    """
    co_located, adjacent = _plunder_threat(ctx)
    if co_located is None and adjacent is None:
        return None
    if on_post:
        return _bank_sell(ctx)
    if _can_shield(ctx):
        return _raise_shields()
    if co_located is not None:
        # Sitting on the raider -> step off, away from it, heading toward a post.
        return _flee_toward_post(ctx, avoid=[(co_located["x"], co_located["y"])])
    # Adjacent raider we can't shield against -> approach a post but never step
    # onto the raider's cell.
    return _flee_toward_post(ctx, avoid=[(adjacent["x"], adjacent["y"])])


# ----------------------------------------------------------------------------
# Balanced miner-trader decision (ported from spaceship.generate_action_response)
# ----------------------------------------------------------------------------
def _balanced_action(ctx):
    # === 1. RESPAWN if destroyed ===
    if ctx.state == "DESTROYED":
        return {"actionType": "RESPAWN"}

    asteroid_here = _entity_at(ctx.x, ctx.y, ctx.asteroids)
    mineable_here = bool(
        asteroid_here and asteroid_here["nutrinium"] > 0 and asteroid_here["mass"] > 0
    )
    target = _find_target_zone(ctx)
    at_target = target is not None and target["x"] == ctx.x and target["y"] == ctx.y
    on_post = _entity_at(ctx.x, ctx.y, ctx.trading_posts) is not None

    # === 2. OPPORTUNISTIC SELL: already on a post with worthwhile cargo ===
    # Selling is free and instant -- never walk off a post holding sellable
    # cargo. (SELL is masked while recharging, so _bank_sell ends the recharge
    # first and the sale lands next tick.)
    if on_post and ctx.nutrinium >= MIN_SELL_QUANTITY:
        return _bank_sell(ctx)

    # === 2b. PLUNDER EVASION: a raider is on/next to us and we hold cargo +++
    # PLUNDER ignores relative energy, so even a weak raider sharing our cell can
    # rob us while shields are DOWN. Bank it, shield up, or step off the raider.
    evade = _evade_plunder(ctx, on_post)
    if evade is not None:
        return evade

    # === 2c. STAND DOWN SHIELDS: no raider left but shields still POWERED ===
    # We only reach here un-threatened (an active raider is handled in 2b). MINE
    # and RECHARGE are masked while shields are POWERED, so drop them now to avoid
    # a struck loop; the sale path above already tolerates shields being up.
    if ctx.shield_state == "POWERED":
        return _lower_shields()

    # === 3. THREAT-AWARE BANKING: protect cargo from a stronger neighbour ===
    if ctx.nutrinium > 0 and _has_potential_attacker(ctx):
        if _entity_at(ctx.x, ctx.y, ctx.trading_posts):
            return _sell(ctx.nutrinium)
        flee = _go_to_post(ctx)
        if flee is not None:
            return flee

    # === 4. MARKET-TIMING SELL ===
    if _is_selling_time(ctx):
        if _entity_at(ctx.x, ctx.y, ctx.trading_posts):
            return _sell(ctx.nutrinium)
        to_post = _go_to_post(ctx)
        if to_post is not None:
            return to_post
        # No post in range; fall through and keep working.

    # === 5. OPPORTUNISTIC ATTACK ===
    prey, commit = _find_prey(ctx)
    if prey is not None:
        return _attack(prey, commit)

    # === 6. MINE vs MOVE (zone valuation) ===
    target = _find_target_zone(ctx)
    asteroid_here = _entity_at(ctx.x, ctx.y, ctx.asteroids)
    if asteroid_here and asteroid_here["nutrinium"] > 0 and asteroid_here["mass"] > 0:
        at_target = target is not None and target["x"] == ctx.x and target["y"] == ctx.y
        ratio = asteroid_here["nutrinium"] / asteroid_here["mass"]
        if at_target or ratio > NUTRINIUM_MIN_RATIO:
            if ctx.energy >= ctx.mine_cost:
                return {"actionType": "MINE"}
            return {"actionType": "RECHARGE"}

    # Travel toward the best zone (jump when far + affordable).
    if target is not None and (target["x"], target["y"]) != (ctx.x, ctx.y):
        return _navigate(ctx, target["x"], target["y"], jump_min_dist=2, jump_margin=15)

    # Nothing rich nearby: bank any meaningful cargo, else explore.
    if ctx.nutrinium >= HAUL_CARGO_THRESHOLD:
        to_post = _go_to_post(ctx)
        if to_post is not None:
            return to_post

    return _move_explore(ctx)


# ----------------------------------------------------------------------------
# Action-mask safety net (shared utils.action_masker)
# ----------------------------------------------------------------------------
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


def _build_mask_state(ctx, masker):
    """Adapt the parsed balanced-bot context into a ``MaskState``.

    The balanced miner-trader never uses shields/modules, so those fields
    default to harmless empty values (those actions stay masked out).
    """
    energy_costs = {
        "mine": ctx.mine_cost,
        "move": ctx.move_cost,
        "attack": ctx.attack_cost,
        "shields": ctx.shields_cost,
        "jump": ctx.jump_unit_cost,
        "plunder": ctx.plunder_cost,
        "negotiate": ctx.negotiate_cost,
    }
    return masker.MaskState(
        x=ctx.x,
        y=ctx.y,
        energy=ctx.energy,
        health=ctx.health,
        nutrinium=ctx.nutrinium,
        credits=ctx.credits,
        destroyed=(ctx.state == "DESTROYED"),
        recharging=ctx.recharging,
        just_recharged=False,
        shield_state="DOWN",
        shield_value=0,
        shield_capacity=0,
        shields_up=False,
        modules=ctx.modules,
        negotiate_post_id=None,
        enemies=ctx.ships,
        asteroids=ctx.asteroids,
        trading_posts=ctx.trading_posts,
        wreckage=[],
        map_width=ctx.map_w,
        map_height=ctx.map_h,
        max_energy=ctx.max_energy,
        max_health=ctx.max_health,
        energy_costs=energy_costs,
        salvage_energy_cost=999,
        repair_cost=0,
        action_restrictions=ctx.action_restrictions,
        jump_min_cost=ctx.jump_min_cost,
        jump_cost_skill=int(_skill(ctx.skills, "jump_cost")),
        max_jump_distance=ctx.max_jump_distance + int(_skill(ctx.skills, "jump_distance")) * 10,
    )


def _action_name_to_id(masker, action, ctx):
    """Map a response dict back to the env action id the masker validates."""
    at = action.get("actionType")
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
    }
    if at in simple:
        return simple[at]
    if at == "MOVE":
        return {
            "N": masker.MOVE_NORTH,
            "S": masker.MOVE_SOUTH,
            "E": masker.MOVE_EAST,
            "W": masker.MOVE_WEST,
        }.get(payload.get("direction"), masker.WAIT)
    if at == "JUMP":
        tgt = payload.get("target_location", {}) or {}
        tx, ty = int(tgt.get("x", ctx.x)), int(tgt.get("y", ctx.y))
        if _entity_at(tx, ty, ctx.trading_posts):
            return masker.JUMP_TO_TRADING_POST
        return masker.JUMP_TO_ASTEROID
    return masker.WAIT


def _richest_zone_enemy(ctx):
    """Richest (by cargo) live enemy sharing our cell, or None."""
    same = _entities_at(ctx.x, ctx.y, ctx.ships)
    if not same:
        return None
    return max(same, key=lambda e: e.get("nutrinium", 0))


def _masked_to_response(ctx, action_id, masker):
    """Rebuild a response dict for an enforced (valid) balanced-bot action id."""
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
    if action_id == masker.MOVE_NORTH:
        return _move("N")
    if action_id == masker.MOVE_SOUTH:
        return _move("S")
    if action_id == masker.MOVE_EAST:
        return _move("E")
    if action_id == masker.MOVE_WEST:
        return _move("W")
    if action_id == masker.SELL:
        return _sell(ctx.nutrinium)
    if action_id == masker.ATTACK:
        enemy = _richest_zone_enemy(ctx)
        if enemy is None:
            return {"actionType": "WAIT"}
        return _attack(enemy, min(DEFAULT_ATTACK_ENERGY, ctx.energy))
    if action_id == masker.JUMP_TO_ASTEROID:
        ast = _nearest_entity(ctx.x, ctx.y, [a for a in ctx.asteroids if a["nutrinium"] > 0])
        return _jump(ast) if ast else {"actionType": "WAIT"}
    if action_id == masker.JUMP_TO_TRADING_POST:
        post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
        return _jump(post) if post else {"actionType": "WAIT"}
    return {"actionType": "WAIT"}


def _enforce(ctx, action):
    """Validate the heuristic's action via the shared masker; substitute if invalid."""
    masker = _get_masker()
    if masker is None:
        return action
    st = _build_mask_state(ctx, masker)
    action_id = _action_name_to_id(masker, action, ctx)
    is_valid, _ = masker.is_action_valid(action_id, st)
    if is_valid:
        return action
    enforced_id = masker.mask_action(action_id, st)
    return _masked_to_response(ctx, enforced_id, masker)


# ----------------------------------------------------------------------------
# Public lambda contract
# ----------------------------------------------------------------------------
def get_heuristic_action(action_request):
    """Decide a balanced miner-trader action for one ActionRequest (raw dict)."""
    ctx = _Context(action_request)
    _update_market_history(ctx.game_id, ctx.game_round, ctx.market_price)
    return _enforce(ctx, _balanced_action(ctx))


def get_action(action_request):
    """Public entry point: returns a normalised response dict."""
    return _to_response(get_heuristic_action(action_request))
