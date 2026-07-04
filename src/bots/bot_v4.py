"""Prospectors & Pirates bot (v4).

A standalone port of the environment's PIRATE opponent AI
(``ProspectorsPiratesEnv._ai_pirate``) onto the same lambda contract as
``bot_v2`` / ``bot_v3``: a single :func:`get_action` that takes an ActionRequest
dict and returns a response dict ``{"actionType": str, "payload"?: ...}``.

The strategy is a pirate-first raider: it tries to acquire nutrinium through
combat (plunder/attack/salvage), holds cargo while market prices are weak, and
only banks when prices are good (or cargo pressure is high):

1. Energy management: short recharge cycles, only when truly low.
2. Pirate priority: PLUNDER (same-zone) -> ATTACK (same-zone) -> SALVAGE.
3. Flee when health is dangerously low and an enemy shares the zone.
4. Sell only when market price is good (or cargo pressure forces a bank).
5. Mine the asteroid under the ship.
6. Carry cargo home only when we actually want to sell now.
7. Otherwise pick the best asteroid by round-trip efficiency and travel to it.

NOTE: unlike the in-environment AI (which has global knowledge of every
asteroid/post/ship on the map), this bot only sees what the live server reports
in ``sensors`` -- entities within the ship's sensor range -- so its choices are
necessarily local. Directions are emitted in the live server frame
(N=y+1, S=y-1, E=x+1, W=x-1), matching the environment's MOVE actions.
"""

import os
import sys


def _to_response(payload):
    """Lightweight lambda response adapter (mirrors bot_v2/bot_v3)."""
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
# Tunables (mirror the reference PIRATE thresholds)
# ----------------------------------------------------------------------------
RECHARGE_END_ENERGY = 70      # stop recharging once back to a workable level
RECHARGE_LOW_ENERGY = 15      # only recharge when truly low (mining turns are precious)
HAUL_CARGO_THRESHOLD = 12     # cargo worth carrying home before topping up
FORCE_SELL_CARGO = 35         # always bank when hold risk is too high
GOOD_SELL_PRICE_FRAC = 0.92   # sell when current price is >= 92% of round high
PLUNDER_MIN_TARGET = 1        # minimum target nutrinium to consider PLUNDER
DEFAULT_ATTACK_ENERGY = 20    # energy spent per ATTACK (matches env default payload)

# Per-round market memory keyed by gameId: used to decide whether the current
# trading-post price is strong enough to bank cargo now.
_market_high = {}    # gameId -> highest sell price seen this round
_market_round = {}   # gameId -> round for the high watermark above


def _update_market_window(ctx):
    """Track the per-round high sell price for this game."""
    if ctx.game_id is None or ctx.market_sell_price is None:
        return
    round_no = int(ctx.game_round or 0)
    if _market_round.get(ctx.game_id) != round_no:
        _market_round[ctx.game_id] = round_no
        _market_high[ctx.game_id] = float(ctx.market_sell_price)
        return
    _market_high[ctx.game_id] = max(float(ctx.market_sell_price),
                                    float(_market_high.get(ctx.game_id, 0.0)))


def _should_sell_now(ctx):
    """Whether we should convert cargo to credits at this tick."""
    if ctx.nutrinium <= 0:
        return False
    if ctx.nutrinium >= FORCE_SELL_CARGO:
        return True
    if ctx.market_sell_price is None:
        return True
    _update_market_window(ctx)
    high = float(_market_high.get(ctx.game_id, ctx.market_sell_price) or 0.0)
    if high <= 0.0:
        return True
    return float(ctx.market_sell_price) >= high * GOOD_SELL_PRICE_FRAC


# ----------------------------------------------------------------------------
# Geometry / parsing helpers (stateless)
# ----------------------------------------------------------------------------
def _distance(x1, y1, x2, y2):
    """Euclidean distance between two cells."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


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


def _direction_towards(dx, dy, tie_horizontal=False):
    """Compass direction to step along the dominant axis (server frame).

    When ``tie_horizontal`` is True a tie (|dx| == |dy|) resolves to E/W,
    matching the PIRATE flee logic; otherwise it resolves to N/S.
    """
    horizontal = abs(dx) >= abs(dy) if tie_horizontal else abs(dx) > abs(dy)
    if horizontal:
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


# ----------------------------------------------------------------------------
# Parsed request
# ----------------------------------------------------------------------------
class _Context:
    """Parses one ActionRequest into the fields the pirate loop needs."""

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
        self.skills = me.get("skills", {}) or {}
        # Installed modules (e.g. JUMP) gate module-locked actions in the masker.
        self.modules = {str(m).upper() for m in (me.get("modules", []) or [])}
        self.player_id = me.get("playerId")

        # Split sensor contacts by type into flat {x, y, ...} dicts.
        self.asteroids = []
        self.trading_posts = []
        self.wreckage = []
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
            elif s_type == "wreckage":
                entry["nutrinium"] = int(s.get("nutrinium", 0))
                self.wreckage.append(entry)
            elif s_type == "ship":
                if str(s.get("state", "")).upper() == "DESTROYED":
                    continue
                entry["health"] = int(s.get("health", 100))
                entry["energy"] = int(s.get("energy", 0))
                entry["nutrinium"] = int(s.get("nutrinium", 0))
                entry["playerId"] = s.get("playerId")
                if entry["playerId"] == self.player_id:
                    continue
                self.ships.append(entry)

        # Jump / combat economics from round metadata.
        game_state = req.get("gameState", {}) or {}
        self.game_id = game_state.get("gameId")
        self.game_round = int(game_state.get("round", 0) or 0)
        metadata = game_state.get("metadata", {}) or {}
        ship_cfg = metadata.get("shipConfig", {}) or {}
        costs = ship_cfg.get("energyCosts", {}) or {}
        self.mine_cost = int(costs.get("mine", 10))
        self.attack_cost = int(costs.get("attack", 1))
        self.salvage_cost = int(costs.get("salvage", 999))
        self.jump_unit_cost = int(costs.get("jump", 1))
        self.jump_min_cost = int(costs.get("jumpMinCost", 75))
        # Extra fields the shared action-masker safety net validates against.
        self.move_cost = int(costs.get("move", 1))
        self.shields_cost = int(costs.get("shields", 5))
        self.plunder_cost = int(costs.get("plunder", 5))
        self.negotiate_cost = int(costs.get("negotiate", 5))
        self.state = str(me.get("state", "READY")).upper()
        self.credits = int(me.get("credits", 0))
        self.max_energy = int(ship_cfg.get("maxEnergy", 100))
        self.max_jump_distance = int(ship_cfg.get("maxJumpDistance", 50))
        self.max_health = int(ship_cfg.get("maxHealth", 100))
        map_cfg = metadata.get("mapConfig", {}) or {}
        self.map_w = int(map_cfg.get("width", 125))
        self.map_h = int(map_cfg.get("height", 125))
        market = metadata.get("market", {}) or {}
        sell_prices = market.get("sell", {}) or {}
        raw_price = sell_prices.get("nutrinium")
        self.market_sell_price = float(raw_price) if raw_price is not None else None
        # Per-state action restrictions (allowedWhileRecharging / allowedWithShieldsUp).
        self.action_restrictions = metadata.get("actionRestrictions", {}) or {}

        # Last-action outcome: when PLUNDER/ATTACK fails (target moved away), back off
        # so we don't waste ticks re-trying the same stale target.
        last_result = req.get("actionResult", {}) or {}
        self.last_action_type = str(last_result.get("actionType", "")).upper()
        self.last_action_failed = str(last_result.get("outcome", "")).upper() == "FAILURE"
        self.offence_target_lost = (
            self.last_action_failed and self.last_action_type in ("PLUNDER", "ATTACK")
        )

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
    # SELL requires a positive-integer ``nutrinium`` payload (game spec:
    # actions.md -> Sell). Callers only reach here with cargo on board.
    return {"actionType": "SELL", "payload": {"nutrinium": int(amount)}}


def _attack(target, energy):
    # ATTACK requires a target ship and an energy payload (game spec:
    # actions.md -> Attack).
    return {"actionType": "ATTACK",
            "payload": {"target": target.get("playerId"), "energy": int(energy)}}


def _plunder(target):
    return {"actionType": "PLUNDER", "payload": {"target": target.get("playerId")}}


def _salvage():
    return {"actionType": "SALVAGE"}


# ----------------------------------------------------------------------------
# Pirate decision (ported from ProspectorsPiratesEnv._ai_pirate)
# ----------------------------------------------------------------------------
def _pirate_action(ctx):
    """Pirate-first raider: plunder/attack/salvage first, sell on good prices."""
    # === 1. ENERGY MANAGEMENT ===
    if ctx.recharging:
        if ctx.energy >= RECHARGE_END_ENERGY:
            return {"actionType": "RECHARGE_END"}
        return {"actionType": "WAIT"}

    if ctx.energy < RECHARGE_LOW_ENERGY:
        return {"actionType": "RECHARGE"}

    _update_market_window(ctx)
    on_post = _entity_at(ctx.x, ctx.y, ctx.trading_posts) is not None

    same_zone_targets = _entities_at(ctx.x, ctx.y, ctx.ships)

    # === 2. PIRATE PRIORITY: PLUNDER then ATTACK then SALVAGE ===
    # Skip if last offence action failed (target moved out of zone).
    if (same_zone_targets and ctx.energy >= ctx.plunder_cost
            and not ctx.offence_target_lost):
        richest = max(same_zone_targets, key=lambda t: t.get("nutrinium", 0))
        if richest.get("nutrinium", 0) >= PLUNDER_MIN_TARGET and ctx.health > 25:
            return _plunder(richest)

    # Same-zone surgical strikes: finish weak ships or punish rich ones.
    # Back off if last PLUNDER/ATTACK failed (target is stale).
    if same_zone_targets and ctx.energy >= ctx.attack_cost and not ctx.offence_target_lost:
        # Find the most profitable target in zone.
        best_value = 0
        best_target = None
        best_target_health = 100
        for t in same_zone_targets:
            t_nutr = t.get("nutrinium", 0)
            t_health = t.get("health", 100)
            # Attack priority:
            # 1) Target at low health (finishable) - always attack
            # 2) Target with valuable cargo even if healthy
            # 3) Weaken any target in zone if we're strong
            if t_health <= 10:
                value = 60 + t_nutr * 3
            elif t_health <= 25:
                value = 40 + t_nutr * 2
            elif t_nutr >= 15:
                value = 25 + t_nutr * 1.5
            elif t_nutr >= 8 and t_health <= 60:
                value = 15 + t_nutr
            elif t_health <= 40:
                value = 10 + t_nutr * 0.5
            else:
                value = 0  # Full-health targets with no cargo not worth it

            if value > best_value:
                best_value = value
                best_target = t
                best_target_health = t_health

        # Attack if target is valuable AND we're healthy enough. The bar is
        # lower for already-weakened targets.
        health_threshold = 30 if best_target_health > 50 else 20
        if best_value > 10 and best_target is not None and ctx.health > health_threshold:
            return _attack(best_target, min(DEFAULT_ATTACK_ENERGY, ctx.energy))

    wreck_here = _entity_at(ctx.x, ctx.y, ctx.wreckage)
    if wreck_here and wreck_here.get("nutrinium", 0) > 0 and ctx.energy >= ctx.salvage_cost:
        return _salvage()

    # === 4. FLEE if health is low and enemies are nearby ===
    if ctx.health < 40 and same_zone_targets:
        threat = same_zone_targets[0]
        dx = ctx.x - threat["x"]
        dy = ctx.y - threat["y"]
        if dx == 0 and dy == 0:
            # Stacked on the threat: head toward the nearest post/asteroid.
            escape = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
            if escape is None:
                escape = _nearest_entity(ctx.x, ctx.y, ctx.asteroids)
            if escape:
                dx = escape["x"] - ctx.x
                dy = escape["y"] - ctx.y
            else:
                dx, dy = 1, 0  # default east
        return _move(_direction_towards(dx, dy, tie_horizontal=True))

    # === 5. SELL only on strong market (or forced by cargo pressure) ===
    if on_post and _should_sell_now(ctx):
        return _sell(ctx.nutrinium)

    # === 6. PRIMARY ECONOMY fallback: mine the asteroid under the ship ===
    asteroid = _entity_at(ctx.x, ctx.y, ctx.asteroids)
    if asteroid and asteroid["nutrinium"] > 0 and ctx.energy >= ctx.mine_cost:
        return {"actionType": "MINE"}

    # === 7. HEAD TO TRADING POST only when we actually want to bank now ===
    if ctx.nutrinium >= HAUL_CARGO_THRESHOLD and _should_sell_now(ctx):
        nearest_post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
        if nearest_post:
            dist = _distance(ctx.x, ctx.y, nearest_post["x"], nearest_post["y"])
            jump_cost = ctx.jump_energy_cost(dist)
            if dist > 1 and ctx.energy >= jump_cost + 5:
                return _jump(nearest_post)
            return _move(_direction_towards(nearest_post["x"] - ctx.x,
                                            nearest_post["y"] - ctx.y))

    # === 8. FIND BEST ASTEROID (prefer rich ones near trading posts) ===
    best_asteroid = None
    best_ast_score = -1.0
    for ast in ctx.asteroids:
        if ast["nutrinium"] <= 0:
            continue
        dist_to_ast = _distance(ctx.x, ctx.y, ast["x"], ast["y"])
        nearest_post = _nearest_entity(ast["x"], ast["y"], ctx.trading_posts)
        dist_ast_to_post = 10.0
        if nearest_post:
            dist_ast_to_post = _distance(ast["x"], ast["y"],
                                         nearest_post["x"], nearest_post["y"])
        total_travel = dist_to_ast + dist_ast_to_post * 0.5
        score = (ast["nutrinium"] ** 1.3) / (total_travel + 1)
        if score > best_ast_score:
            best_ast_score = score
            best_asteroid = ast

    if best_asteroid:
        dist = _distance(ctx.x, ctx.y, best_asteroid["x"], best_asteroid["y"])
        jump_cost = ctx.jump_energy_cost(dist)
        if dist > 2 and ctx.energy >= jump_cost + 15:
            return _jump(best_asteroid)
        return _move(_direction_towards(best_asteroid["x"] - ctx.x,
                                        best_asteroid["y"] - ctx.y))

    return {"actionType": "WAIT"}


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
    """Adapt the parsed pirate context into a ``MaskState``.

    A pirate mines, sells and strikes weak ships, but never uses
    shields/modules, so those fields default to harmless empty values.
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
        wreckage=ctx.wreckage,
        map_width=ctx.map_w,
        map_height=ctx.map_h,
        max_energy=ctx.max_energy,
        max_health=ctx.max_health,
        energy_costs=energy_costs,
        salvage_energy_cost=ctx.salvage_cost,
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
        "SALVAGE": masker.SALVAGE,
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
    """Rebuild a response dict for an enforced (valid) pirate action id."""
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
    if action_id == masker.PLUNDER:
        enemy = _richest_zone_enemy(ctx)
        if enemy is None:
            return {"actionType": "WAIT"}
        return _plunder(enemy)
    if action_id == masker.SALVAGE:
        return _salvage()
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
    """Decide a pirate action for one ActionRequest (raw dict, pre-adapter)."""
    ctx = _Context(action_request)
    return _enforce(ctx, _pirate_action(ctx))


def get_action(action_request):
    """Public entry point: returns a normalised response dict."""
    return _to_response(get_heuristic_action(action_request))
