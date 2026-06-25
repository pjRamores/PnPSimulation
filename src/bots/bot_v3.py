"""Prospectors & Pirates bot (v3).

A standalone port of the environment's PROSPECTOR opponent AI
(``ProspectorsPiratesEnv._ai_prospector``) onto the same lambda contract as
``bot_v2``: a single :func:`get_action` that takes an ActionRequest dict and
returns a response dict ``{"actionType": str, "payload"?: ...}``.

The strategy is a pure mining/selling economy loop -- never fights:

1. Energy management: short recharge cycles, only when truly low.
2. Sell any cargo at the current trading post.
3. Carry cargo (>=12) home: jump aggressively, else walk.
4. Mine the asteroid under the ship.
5. Otherwise pick the best asteroid by round-trip efficiency and travel to it.
6. Sell leftover cargo when there is nothing worth mining.

NOTE: unlike the in-environment AI (which has global knowledge of every
asteroid/post on the map), this bot only sees what the live server reports in
``sensors`` -- entities within the ship's sensor range -- so its choices are
necessarily local. Directions are emitted in the live server frame
(N=y+1, S=y-1, E=x+1, W=x-1), matching the environment's MOVE actions.
"""

import os
import sys


def _to_response(payload):
    """Lightweight lambda response adapter (mirrors bot_v2)."""
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
# Tunables (mirror the reference PROSPECTOR thresholds)
# ----------------------------------------------------------------------------
RECHARGE_END_ENERGY = 60     # stop recharging once back to a workable level
RECHARGE_LOW_ENERGY = 15     # only recharge when truly low (mining turns are precious)
HAUL_CARGO_THRESHOLD = 12    # cargo worth carrying home before topping up


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


# ----------------------------------------------------------------------------
# Parsed request
# ----------------------------------------------------------------------------
class _Context:
    """Parses one ActionRequest into the fields the prospector loop needs."""

    def __init__(self, action_request):
        req = action_request or {}
        me = req.get("me", {}) or {}
        loc = me.get("location", {}) or {}
        self.x = int(loc.get("x", 0))
        self.y = int(loc.get("y", 0))
        self.energy = int(me.get("energy", 0))
        self.nutrinium = int(me.get("nutrinium", 0))
        self.recharging = bool(me.get("recharging", False))
        self.skills = me.get("skills", {}) or {}
        # Installed modules (e.g. JUMP) gate module-locked actions in the masker.
        self.modules = {str(m).upper() for m in (me.get("modules", []) or [])}

        # Split sensor contacts by type into flat {x, y, ...} dicts.
        self.asteroids = []
        self.trading_posts = []
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

        # Jump economics from round metadata.
        metadata = (req.get("gameState", {}) or {}).get("metadata", {}) or {}
        ship_cfg = metadata.get("shipConfig", {}) or {}
        costs = ship_cfg.get("energyCosts", {}) or {}
        self.mine_cost = int(costs.get("mine", 10))
        self.jump_unit_cost = int(costs.get("jump", 1))
        self.jump_min_cost = int(costs.get("jumpMinCost", 75))
        # Extra fields the shared action-masker safety net validates against.
        self.move_cost = int(costs.get("move", 1))
        self.attack_cost = int(costs.get("attack", 1))
        self.shields_cost = int(costs.get("shields", 5))
        self.plunder_cost = int(costs.get("plunder", 5))
        self.negotiate_cost = int(costs.get("negotiate", 5))
        self.state = str(me.get("state", "READY")).upper()
        self.health = int(me.get("health", 100))
        self.credits = int(me.get("credits", 0))
        self.max_energy = int(ship_cfg.get("maxEnergy", 100))
        self.max_health = int(ship_cfg.get("maxHealth", 100))
        map_cfg = metadata.get("mapConfig", {}) or {}
        self.map_w = int(map_cfg.get("width", 125))
        self.map_h = int(map_cfg.get("height", 125))
        # Per-state action restrictions (allowedWhileRecharging / allowedWithShieldsUp).
        self.action_restrictions = metadata.get("actionRestrictions", {}) or {}

    def jump_energy_cost(self, distance):
        """Energy cost of a jump of the given distance (skill lowers the floor)."""
        adj_min_cost = max(0, self.jump_min_cost - _skill(self.skills, "jump_cost") * 5)
        return int(max(adj_min_cost, round(self.jump_unit_cost * distance)))


# ----------------------------------------------------------------------------
# Prospector decision (ported from ProspectorsPiratesEnv._ai_prospector)
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


def _prospector_action(ctx):
    """Optimised mining-selling loop with efficient travel (never fights)."""
    # === 1. ENERGY MANAGEMENT ===
    if ctx.recharging:
        # Short recharge: get back to work quickly.
        if ctx.energy >= RECHARGE_END_ENERGY:
            return {"actionType": "RECHARGE_END"}
        return {"actionType": "WAIT"}

    # Recharge only when truly low -- every recharge turn is a lost mining turn.
    if ctx.energy < RECHARGE_LOW_ENERGY:
        return {"actionType": "RECHARGE"}

    # === 2. SELL at trading post -- always, any amount ===
    if ctx.nutrinium > 0 and _entity_at(ctx.x, ctx.y, ctx.trading_posts):
        return _sell(ctx.nutrinium)

    # === 3. HEAD TO TRADING POST when carrying cargo ===
    if ctx.nutrinium >= HAUL_CARGO_THRESHOLD:
        nearest_post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
        if nearest_post:
            dist = _distance(ctx.x, ctx.y, nearest_post["x"], nearest_post["y"])
            jump_cost = ctx.jump_energy_cost(dist)
            # Jump to trading post aggressively (even short distances).
            if dist > 1 and ctx.energy >= jump_cost + 5:
                return _jump(nearest_post)
            # Walk to trading post.
            return _move(_direction_towards(nearest_post["x"] - ctx.x,
                                            nearest_post["y"] - ctx.y))

    # === 4. MINE current asteroid ===
    asteroid = _entity_at(ctx.x, ctx.y, ctx.asteroids)
    if asteroid and asteroid["nutrinium"] > 0 and ctx.energy >= ctx.mine_cost:
        return {"actionType": "MINE"}

    # === 5. FIND BEST ASTEROID (optimised for round-trip efficiency) ===
    best_asteroid = None
    best_score = -1.0
    for ast in ctx.asteroids:
        if ast["nutrinium"] <= 0:
            continue
        dist_to_ast = _distance(ctx.x, ctx.y, ast["x"], ast["y"])
        # Factor in distance from asteroid to nearest trading post.
        nearest_post = _nearest_entity(ast["x"], ast["y"], ctx.trading_posts)
        dist_to_post = 10.0
        if nearest_post:
            dist_to_post = _distance(ast["x"], ast["y"],
                                     nearest_post["x"], nearest_post["y"])
        # Round-trip cost: getting there + getting to a trading post after.
        round_trip = dist_to_ast + dist_to_post * 0.6
        # Favour rich asteroids (nutrinium^1.3), penalise by round-trip distance.
        score = (ast["nutrinium"] ** 1.3) / (round_trip + 1)
        if score > best_score:
            best_score = score
            best_asteroid = ast

    if best_asteroid:
        dist = _distance(ctx.x, ctx.y, best_asteroid["x"], best_asteroid["y"])
        jump_cost = ctx.jump_energy_cost(dist)
        # Jump aggressively -- walking wastes turns.
        if dist > 2 and ctx.energy >= jump_cost + 10:
            return _jump(best_asteroid)
        # Walk towards asteroid.
        return _move(_direction_towards(best_asteroid["x"] - ctx.x,
                                        best_asteroid["y"] - ctx.y))

    # === 6. SELL remaining cargo if nothing to mine ===
    if ctx.nutrinium > 0:
        nearest_post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
        if nearest_post:
            dist = _distance(ctx.x, ctx.y, nearest_post["x"], nearest_post["y"])
            jump_cost = ctx.jump_energy_cost(dist)
            if dist > 1 and ctx.energy >= jump_cost + 5:
                return _jump(nearest_post)
            return _move(_direction_towards(nearest_post["x"] - ctx.x,
                                            nearest_post["y"] - ctx.y))

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
    """Adapt the parsed prospector context into a ``MaskState``.

    A prospector never fights or uses shields/modules, so combat/shield/module
    fields default to harmless empty values (those actions stay masked out).
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
        enemies=[],
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
        "RESPAWN": masker.RESPAWN,
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


def _masked_to_response(ctx, action_id, masker):
    """Rebuild a response dict for an enforced (valid) prospector action id."""
    if action_id == masker.MINE:
        return {"actionType": "MINE"}
    if action_id == masker.RECHARGE:
        return {"actionType": "RECHARGE"}
    if action_id == masker.RECHARGE_END:
        return {"actionType": "RECHARGE_END"}
    if action_id == masker.RESPAWN:
        return {"actionType": "RESPAWN"}
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
    """Decide a prospector action for one ActionRequest (raw dict, pre-adapter)."""
    ctx = _Context(action_request)
    return _enforce(ctx, _prospector_action(ctx))


def get_action(action_request):
    """Public entry point: returns a normalised response dict."""
    return _to_response(get_heuristic_action(action_request))
