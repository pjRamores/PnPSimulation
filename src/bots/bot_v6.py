"""
Prospectors & Pirates bot (v6) -- model-backed.

Unlike the pure-stdlib heuristic bots (`'bot_v2'` .. `'bot_v5'`), this bot
delegates the decision to a trained reinforcement-learning model. It mirrors the
same 2026 lambda contract -- a single :func:`get_action` that takes an
ActionRequest dict and returns `"{actionType": str, "payload": ...}` -- but
the action is chosen by running the model over a reconstructed observation.

How it works:

1. :class:`_Context` parses the ActionRequest (`'me'` / `'sensors'` /
   `'gameState.metadata'`) into the fields the environment's observation builder
   consumes.
2. :func:`_build_observation` reconstructs the environment's 224-dim observation
   vector following the exact layout of `env_observation_mixin._get_observation`.
   The "global" segments (top asteroids, nearest trading post, enemy extremes,
   nearest wreckage) are reconstructed from SENSOR-VISIBLE entities only -- the
   lambda only sees what the server reports in `'sensors'` (entities within
   sensor range), so these are necessarily local/myopic compared to the
   in-environment AIs that enjoy global map knowledge.
3. :func:`_mask_state` adapts the parsed state into a
   `utils.action_masker.MaskState` and the shared action-masking utility
   builds the env's 19-action validity mask (single source of truth with the
   environment) over SENSOR-VISIBLE entities only.
4. The model predicts an action; its raw action TYPE is run through
   `action_masker.mask_action` so the bot only ever commits to a valid action
   (the same enforcement the environment applies to the player). The model's own
   target coordinate / energy bin are reused for JUMP / ATTACK payloads.
5. :func:`_action_to_response` converts the chosen env action id back into a
   lambda response dict (the inverse of the environment's `_translate_bot_action`).

The default model is configurable via :data:`MODEL_NAME` (resolved relative to
`../models/<MODEL_NAME>`). Any failure to import numpy / stable_baselines3 or
to load the model degrades gracefully to `{"actionType": "WAIT"}`.
"""

import math
import os
import sys

try:
    import numpy as np
    _NUMPY_OK = True
except Exception:  # pragma: no cover - numpy is expected but degrade gracefully
    np = None
    _NUMPY_OK = False


# Configuration (editable)
# ---------------------------------
# Default model file under `../models/` (the `.zip` suffix is optional -- stable_baselines3 appends it). Point this at any current full-spec model (# 224-dim Dict observation, 19 action types). `ppo_pnp_model_v9` is a native Multidiscrete([19, map_w, map_h, energy_bins]) model that matches the env.
MODEL_NAME = "ppo_pnp_model_v9"

# Observation / action format the loaded model expects, resolved via
# `model_specs.get_named_model_spec`. Supported names: `'FULL'` (224-dim,
# the default), `'COMPACT'` (20-dim essentials) and `'SENSOR_ONLY'`
# (6 + (2*sensor_range+1)^2 local grid). The matching observation is
# reconstructed from sensor-visible entities. Unknown names fall back to FULL.
MODEL_SPEC = "FULL"

_NUM_ACTIONS = 19
_OBS_SIZE = 224
_ENERGY_BINS = 11  # MultiDiscrete energy dimension (matches the env)
_DEFAULT_ATTACK_ENERGY = 20  # baseline ATTACK payload when the model gives none

# Environment config constants that the observation normalizers depend on but
# that the ActionRequest does not carry (they match the simulator's config).
_MAX_CARGO = 1000  # max_nutrinium_cargo
_MAX_CREDITS = 10000  # max_credits
_MAX_HEALTH = 100  # max_health
_MAX_SKILL_POINTS = 24  # max_skill_points
_TOP_ASTEROIDS_COUNT = 5  # top_asteroids_count
_MAX_STEPS = 300  # episode length proxy for the action counter
_MARKET_REF = 98.0  # config market.sell_nutrinium reference price

# Action id -> name (matches env_common.ActionType, display/translate order).
_ACTION_NAMES = [
    "WAIT", "MINE", "MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST",
    "RECHARGE", "RECHARGE_END", "ATTACK", "JUMP_TO_ASTEROID", "SELL",
    "RAISE_SHIELDS", "JUMP_TO_TRADING_POST", "RESPAWN", "PLUNDER", "SALVAGE",
    "REPAIR", "NEGOTIATE", "LOWER_SHIELDS",
]

# Simple env action ids that translate to a bare `{"actionType": NAME}`.
_SIMPLE_ACTIONS = {
    "WAIT", "MINE", "RECHARGE", "RECHARGE_END", "RAISE_SHIELDS",
    "LOWER_SHIELDS", "RESPAWN", "PLUNDER", "SALVAGE", "REPAIR", "NEGOTIATE",
}

def _restriction_key(action_name):
    """Map an action name to its `metadata.actionRestrictions` key.
    The four MOVE directions share the `MOVE` rule and both JUMP variants share
    the `JUMP` rule; every other action maps to itself. Mirrors
    `utils.action_masker.ACTION_RESTRICTION_NAME`.
"""
if action_name.startswith("MOVE_"):
    return "MOVE"
if action_name.startswith("JUMP_"):
    return "JUMP"
return action_name

# Action id -> restriction key, aligned to the 19-action mask order.
_ACTION_RESTRICTION_KEY = [_restriction_key(n) for n in _ACTION_NAMES]

def to_response(payload):
    """Lightweight lambda response adapter (mirrors bot_v2 .. bot_v5)."""
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

# ---------------------------------
# Lazy model loading (cached at module scope)
# ---------------------------------
_MODEL = "unset"  # sentinel -> (model, is_multidiscrete) tuple, or None on failure

def load_model():
    """Load and cache the configured model. Returns `(model, is_md)` or `None`.

    The parent ``PnPSimulation`` directory is added to ``sys.path`` so the
    model and its dependencies resolve. Any failure (missing numpy / sb3 / model file)
    is swallowed and cached as ``None`` so callers fall back to WAIT without re-attempting a slow load every tick.
    """
    global _MODEL
    if _MODEL != "unset":
        return _MODEL

    result = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(here)  # ../PnPSimulation
        if parent not in sys.path:
            sys.path.insert(0, parent)
        model_path = os.path.join(parent, "models", MODEL_NAME)

        from stable_baselines3 import PPO
        try:
            from gymnasium import spaces
        except Exception:
            spaces = None

        model = PPO.load(model_path)
        if spaces is not None:
            is_md = isinstance(model.action_space, spaces.MultiDiscrete)
        else:
            is_md = hasattr(model.action_space, "nvec")
        result = (model, is_md)
    except Exception:
        result = None

    _MODEL = result
    return result


_MASKER = "unset"  # sentinel -> the utils.action_masker module, or None on failure

def get_masker():
    """Lazily import and cache `utils.action_masker`. Returns the module or None.

    The ``src`` directory (parent of ``bots``) is added to ``sys.path`` so the
    utility resolves both inside the simulator and when run as a standalone
    lambda. Any failure degrades gracefully (callers fall back to no masking).
    """
    global _MASKER
    if _MASKER != "unset":
        return _MASKER
    result = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(here)  # ../src
        if src_dir not in sys.path:
sys.path.insert(0, src_dir)
from utils import action_masker
result = action_masker
except Exception:
    result = None
MASKER = result
return result

_SPEC = "unset"  # sentinel -> resolved model_specs.ModelSpec, or None on failure

def _load_spec():
    """Resolve and cache the :data:`MODEL_SPEC` into a `model_specs.ModelSpec`.

    The ``src`` directory (parent of ``bots``) is added to ``sys.path`` so
    ``model_specs`` resolves both inside the simulator and as a standalone
    lambda. An unknown spec name falls back to ``DEFAULT_FULL_SPEC``; any import
    failure caches ``None`` (the caller then defaults to the full observation).
    """
    global _SPEC
    if _SPEC != "unset":
        return _SPEC
    result = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(here)  # .../src
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from model_specs import get_named_model_spec, DEFAULT_FULL_SPEC
        result = get_named_model_spec(MODEL_SPEC) or DEFAULT_FULL_SPEC
    except Exception:
        result = None
    _SPEC = result
    return result

# ---------------------------------------------------------
# Geometry / parsing helpers (stateless)

def distance(x1, y1, x2, y2):
    """Euclidean distance between two cells."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def entity_at(x, y, entities):
    """First entity exactly at (x, y), or None."""
    for e in entities:
        if e["x"] == x and e["y"] == y:
            return e
    return None

def entities_at(x, y, entities):
    """All entities exactly at (x, y)."""
    return [e for e in entities if e["x"] == x and e["y"] == y]

def nearest_entity(x, y, entities):
    """Nearest entity to (x, y) by Euclidean distance, or None."""
    best = None
    best_dist = None
    for e in entities:
        d = _distance(x, y, e["x"], e["y"])
        if best_dist is None or d < best_dist:
            best_dist = d
            best = e
    return best

def ab(skills, name, default):
    """Skill/ability value (numeric), falling back to `default`."""
    try:
        v = skills.get(name, default)
        return float(v if v is not None else default)
    except (AttributeError, TypeError, ValueError):
        return float(default)

# ---------------------------------------------------------
# Parsed request

class _Context:
    """Parses one ActionRequest into the fields the observation builder needs."""

    def __init__(self, action_request):
        req = action_request or {}
        me = req.get("me", {}) or {}
        loc = me.get("location", {}) or {}
        self.x = int(loc.get("x", 0))
        self.y = int(loc.get("y", 0))
        self.energy = int(me.get("energy", 0))
        self.health = int(me.get("health", _MAX_HEALTH))
        self.nutrinium = int(me.get("nutrinium", 0))
        self.credits = int(me.get("credits", 0) or 0)
        self.recharging = bool(me.get("recharging", False))
        self.state = str(me.get("state", "READY")).upper()
        self.skills = me.get("skills", {}) or {}
        self.player_id = me.get("playerId")
        self.skill_points_total = int(me.get("skillPointsTotal", 5) or 5)
self.skill_points_spent = int(me.get("skillPointsSpent", 0) or 0)

tid = me.get("teamId")
self.team_id = int(tid) if tid is not None else 0

shield = me.get("shield", {}) or {}
self._shield_state = str(shield.get("state", "DOWN") or "DOWN").upper()
self.shield_value = float(shield.get("value", 0) or 0)
self.shield_capacity = float(shield.get("capacity", 0) or 0)
self.shields_up = self._shield_state == "POWERED"

self.modules = set()
for m in me.get("modules", []) or []:
    if isinstance(m, str):
        self.modules.add(m.upper())
    elif isinstance(m, dict):
        tag = m.get("type") or m.get("name")
        if tag:
            self.modules.add(str(tag).upper())

objective = (me.get("objectives") or {}).get("negotiate") or {}
self.negotiate_post_id = objective.get("tradingPostId")

# Split sensor contacts by type into flat {x, y, ...} dicts.
self.asteroids = []
self.trading_posts = []
self.ships = []
self.wreckage = []

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
        entry["health"] = int(s.get("health", MAX_HEALTH))
        entry["energy"] = int(s.get("energy", 0))
        entry["credits"] = int(s.get("credits", 0))
        entry["nutrinium"] = int(s.get("nutrinium", 0))
        entry["playerId"] = s.get("playerId")
        entry["skills"] = s.get("skills", {}) or {}
        sh = s.get("shield", {}) or {}
        entry["shield_state"] = str(sh.get("state", "") or "").upper()
        self.ships.append(entry)

# Round metadata: economics + map size + market price.
game_state = req.get("gameState", {}) or {}
self.game_id = game_state.get("gameId")
self.game_round = game_state.get("round", 0)
self.tick = int(game_state.get("tick", 0) or 0)
metadata = game_state.get("metadata", {}) or {}
ship_cfg = metadata.get("shipConfig", {}) or {}
costs = ship_cfg.get("energyCosts", {}) or {}
self.mine_cost = int(costs.get("mine", 10))
self.move_cost = int(costs.get("move", 0))
self.attack_cost = int(costs.get("attack", 1))
self.shields_cost = int(costs.get("shields", 1))
self.plunder_cost = int(costs.get("plunder", 5))
self.negotiate_cost = int(costs.get("negotiate", 5))
self.jump_unit_cost = int(costs.get("jump", 1))
self.jump_min_cost = int(costs.get("jumpMinCost", 75))
self.max_energy = int(ship_cfg.get("maxEnergy", 100))
self.per_recharge = int(ship_cfg.get("energyPerRecharge", 10))
sensors_cfg = ship_cfg.get("sensors", {}) or {}
self.sensor_range = int(sensors_cfg.get("range", 5))
map_cfg = metadata.get("mapConfig", {}) or {}
self.map_w = int(map_cfg.get("width", 10))
self.map_h = int(map_cfg.get("height", 10))
self.asteroid_mass_max = float(map_cfg.get("maxMass", 80))
combat = metadata.get("combat", {}) or {}
self.base_shield_capacity = float(combat.get("baseShieldCapacity", 100))
salvage = metadata.get("salvage", {}) or {}
self.salvage_cost = int(salvage.get("energyCost", 5))
market = metadata.get("market", {}) or {}
sell_cfg = market.get("sell", {}) or {}
price = sell_cfg.get("nutrinium")
self.market_price = float(price) if price is not None else 0.0
# Per-state action restrictions (allowedWhileRecharging / allowedWithShieldsUp).
self.action_restrictions = metadata.get("actionRestrictions", {}) or {}

# Convenience constants (config-derived, not in the request).
self.max_health = MAX_HEALTH
self.max_cargo = _MAX_CARGO
self.max_credits = _MAX_CREDITS
self.max_skill_points = _MAX_SKILL_POINTS
self.top_count = _TOP_ASTEROIDS_COUNT
self.max_steps = _MAX_STEPS

def jump_energy_cost(self, distance):
def _top_asteroids(ctx, x, y, count):
    """Top-N asteroids by the env's score (concentration * nutrinium / dist).
    
    Mirrors `EnvObservationMixin._get_top_asteroids` but over sensor-visible 
    asteroids only.
    """
    scored = []
    max_score = 50.0
    for a in ctx.asteroids:
        nutr = a.get("nutrinium", 0)
        if nutr <= 0:
            continue
        dist = _distance(x, y, a["x"], a["y"])
        mass = max(1, a.get("mass", 1))
        concentration = nutr / mass
        raw = concentration * nutr / (dist + 1)
        scored.append({
            "x": a["x"],
            "y": a["y"],
            "mass": a.get("mass", 0),
            "nutrinium": nutr,
            "distance": dist,
            "score": min(1.0, raw / max_score),
        })
    scored.sort(key=lambda e: e["score"], reverse=True)
    return scored[count]

def _combat_score(enemy, raw=False):
    """Weighted enemy combat score (mirrors `_calculate_enemy_combat_score`)."""
    skills = enemy.get("skills", {}) or {}
    score = (
        enemy.get("health", 0) * 1.0
        + enemy.get("energy", 0) * 0.5
        + enemy.get("credits", 0) * 0.1
        + _ab(skills, "attack_power", 0) * 10.0
        + _ab(skills, "attack_accuracy", 0) * 5.0
        + _ab(skills, "shield_strength", 0) * 8.0
        + _ab(skills, "evade", 0) * 3.0
    )
    if raw:
        return score
    return min(1.0, score / 510.0)

def extreme_enemies(ctx):
    """Strongest / weakest visible enemy (mirrors `_get_extreme_enemies`)."""
    active = list(ctx.ships)
    if not active:
        return None, None
    if len(active) == 1:
        return active[0], active[0]
    scored = sorted(active, key=lambda e: _combat_score(e, raw=True), reverse=True)
    return scored[0], scored[-1]

def build_observation(ctx, spec=None):
    """Dispatch to the observation builder matching `spec`'s observation type.
    
    Supports `full` (224-dim), `compact` (20-dim) and `sensor_only`
    (6 + (2*sensor_range+1)^2). Anything else (or a missing spec) builds 
    the full observation.
    """
    obs_type = "full"
    if spec is not None:
        try:
            obs_type = spec.observation_spec.observation_type
        except Exception:
            obs_type = "full"
    if obs_type == "compact":
        return build_compact_observation(ctx)
    if obs_type == "sensor_only":
        return build_sensor_only_observation(ctx)
    return _build_full_observation(ctx)

def _build_full_observation(ctx):
    """Reconstruct the env's 224-dim observation vector from the request."""
    o = []
    sk = ctx.skills

    # === ENHANCED SHIP STATE (24 values) ===
    o.append(ctx.x / max(1, ctx.map_w))
    o.append(ctx.y / max(1, ctx.map_h))
    o.append(ctx.energy / max(1, ctx.max_energy))
    o.append(ctx.health / max(1, ctx.max_health))
    o.append(min(ctx.nutrinium, ctx.max_cargo) / max(1, ctx.max_cargo))
    o.append(min(ctx.credits, ctx.max_credits) / max(1, ctx.max_credits))
    o.append(1.0 if ctx.recharging else 0.0)
    o.append(1.0 if ctx.shields_up else 0.0)
    o.append(1.0 if ctx.state == "READY" else 0.0)
    o.append(ctx.skill_points_total / max(1, ctx.max_skill_points))
o.append(ctx.skill_points_spent / max(1, ctx.max_skill_points)))
o.append(_ab(sk, "energy_max", 5) / 10.0)
o.append(_ab(sk, "recharge_energy", 0) / 10.0)
o.append(_ab(sk, "mine_accuracy", 0) / 10.0)
o.append(_ab(sk, "mine_yield_multiplier", 1) / 5.0)
o.append(_ab(sk, "mine_cost", 2) / 10.0)
o.append(_ab(sk, "combat_salvage_multiplier", 0) / 5.0)
o.append(_ab(sk, "sensor_range", 1) / max(1, ctx.sensor_range))
o.append(_ab(sk, "attack_accuracy", 0) / 10.0)
o.append(_ab(sk, "attack_power", 0) / 10.0)
o.append(_ab(sk, "evade", 0) / 10.0)
o.append(_ab(sk, "shield_strength", 0) / 10.0)
o.append(_ab(sk, "jump_distance", 0) / 10.0)
o.append(ctx.tick / max(1, ctx.max_steps))

# === STRATEGIC CONTEXT (8 values) ===
ast_here = _entity_at(ctx.x, ctx.y, ctx.asteroids)
o.append(1.0 if (ast_here and ast_here["nutrinium"] > 0) else 0.0)
tp_here = _entity_at(ctx.x, ctx.y, ctx.trading_posts)
o.append(1.0 if tp_here else 0.0)
o.append(min(1.0, ctx.nutrinium / 25.0))
enemy_here = _entity_at(ctx.x, ctx.y, ctx.ships) is not None
o.append(1.0 if enemy_here else 0.0)
top1 = _top_asteroids(ctx, ctx.x, ctx.y, 1)
if top1:
    o.append((top1[0]["x"] - ctx.x) / max(1, ctx.map_w))
    o.append((top1[0]["y"] - ctx.y) / max(1, ctx.map_h))
else:
    o.extend([0.0, 0.0])
nearest_tp = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
if nearest_tp:
    o.append((nearest_tp["x"] - ctx.x) / max(1, ctx.map_w))
    o.append((nearest_tp["y"] - ctx.y) / max(1, ctx.map_h))
else:
    o.extend([0.0, 0.0])

# === LOCAL SENSOR GRID (11x11 = 121 values) ===
sr = ctx.sensor_range
side = 2 * sr + 1
x_min = ctx.x - sr
y_min = ctx.y - sr
x_min = max(0, min(x_min, ctx.map_w - side)) if ctx.map_w >= side else 0
y_min = max(0, min(y_min, ctx.map_h - side)) if ctx.map_h >= side else 0
for row in range(side):
    for col in range(side):
        x = x_min + col
        y = y_min + row
        if 0 <= x < ctx.map_w and 0 <= y < ctx.map_h:
            if x == ctx.x and y == ctx.y:
                o.append(0.0)
            elif _entity_at(x, y, ctx.ships):
                o.append(1.0)
            elif _entity_at(x, y, ctx.trading_posts):
                o.append(0.66)
            elif _entity_at(x, y, ctx.asteroids):
                o.append(0.33)
            else:
                o.append(0.0)
        else:
            o.append(-1.0)

# === TOP 5 ASTEROIDS (30 values) ===
top5 = _top_asteroids(ctx, ctx.x, ctx.y, ctx.top_count)
max_dist = math.sqrt(ctx.map_w ** 2 + ctx.map_h ** 2)
max_mass = max(1.0, ctx.asteroid_mass_max)
for a in top5:
    o.append(a["x"] / max(1, ctx.map_w))
    o.append(a["y"] / max(1, ctx.map_h))
    o.append(a["mass"] / max_mass)
    o.append(a["nutrinium"] / max_mass)
    o.append(a["distance"] / max(1.0, max_dist))
    o.append(a["score"])
for _ in range(ctx.top_count - len(top5)):
    o.extend([0.0, 0.0, 0.0, 0.0, 0.0])

# === NEAREST TRADING POST (3 values) ===
if nearest_tp:
    dist = _distance(ctx.x, ctx.y, nearest_tp["x"], nearest_tp["y"])
    o.append(nearest_tp["x"] / max(1, ctx.map_w))
    o.append(nearest_tp["y"] / max(1, ctx.map_h))
    o.append(dist / max(1.0, max_dist))
else:
    o.extend([0.0, 0.0, 0.0])

# === TWO ENEMY TYPES (14 values) ===
strongest, weakest = _extreme_enemies(ctx)
for enemy in (strongest, weakest):
    if enemy:
        o.append(enemy["x"] / max(1, ctx.map_w))
        o.append(enemy["y"] / max(1, ctx.map_h))
        o.append(enemy["energy"] / max(1, ctx.max_energy))
        o.append(enemy["health"] / max(1, ctx.max_health))
        o.append(min(enemy["nutrinium"], 100) / 100.0)
        o.append(min(enemy["credits"], 1000) / 1000.0)
        o.append(_combat_score(enemy))
    else:
        o.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# === SPEC-FIDELITY FEATURES (24 values) ===
o.append(_ab(sk, "shield_capacity", 0) / 10.0)

o.append(ab(sk, "shield_efficiency", 0) / 10.0)
o.append(ab(sk, "jump_cost", 0) / 10.0)
o.append(ab(sk, "salvage_yield", 0) / 10.0)
o.append(ab(sk, "negotiate_skill", 0) / 10.0)
o.append(ab(sk, "negotiate_cautious", 0) / 10.0)
o.append(ab(sk, "negotiate_ambition", 0) / 10.0)

scap = ctx.shield_capacity
sval = ctx.shield_value
max_capacity = ctx.base_shield_capacity + 10 * 10  # base + max_abil shield_capacity*10
o.append(1.0 if ctx._shield_state == "POWERED" else 0.0)
o.append(1.0 if ctx._shield_state == "DRAINING" else 0.0)
o.append(1.0 if ctx._shield_state == "DOWN" else 0.0)
o.append((sval / scap) if scap > 0 else 0.0)
o.append(scap / max(1.0, max_capacity))

o.append(1.0 if "JUMP" in ctx.modules else 0.0)
o.append(1.0 if "REPAIR" in ctx.modules else 0.0)
o.append(1.0 if "SALVAGE" in ctx.modules else 0.0)

o.append(min(1.0, ctx.team_id / 3.0))
o.append(0.0)  # team_bonus is not exposed in the request
price = ctx.market_price if ctx.market_price > 0 else _MARKET_REF
o.append(min(1.0, price / _MARKET_REF))

obj_post = None
if ctx.negotiate_post_id is not None:
    obj_post = next(
        (p for p in ctx.trading_posts if p.get("id") == ctx.negotiate_post_id),
        None,
    )
if obj_post:
    o.append(1.0)
    o.append((obj_post["x"] - ctx.x) / max(1, ctx.map_w))
    o.append((obj_post["y"] - ctx.y) / max(1, ctx.map_h))
else:
    o.extend([0.0, 0.0, 0.0])

nearest_wreck = None
if ctx.wreckage:
    nearest_wreck = min(
        ctx.wreckage,
        key=lambda w: (w["x"] - ctx.x) ** 2 + (w["y"] - ctx.y) ** 2,
    )
if nearest_wreck:
    o.append(1.0)
    o.append((nearest_wreck["x"] - ctx.x) / max(1, ctx.map_w))
    o.append((nearest_wreck["y"] - ctx.y) / max(1, ctx.map_h))
else:
    o.extend([0.0, 0.0, 0.0])

# Action restrictions (38 values: 19 actions * 2 flags), mirroring the env's
# _action_restriction_features. Encodes the request's actionRestrictions matrix
# so a restriction-aware model can adapt; missing rules default to allowed.
restrictions = ctx.action_restrictions or {}
for key in ACTION_RESTRICTION_KEY:
    rule = restrictions.get(key, ()) if key is not None else {}
    o.append(1.0 if rule.get("allowedWhileRecharging", True) else 0.0)
    o.append(1.0 if rule.get("allowedWithShieldsUp", True) else 0.0)

return np.array(o, dtype=np.float32)

def build_compact_observation(ctx):
    """Reconstruct the 20-dim compact observation (mirrors `CompactObservationGenerator`).
    Layout: ship state (8) + nearest asteroid (4) + nearest trading post (3) +
    first visible enemy (5), over sensor-visible entities only.
    """
    map_span = max(1, ctx.map_w + ctx.map_h)
    o = [
        ctx.x / max(1, ctx.map_w),
        ctx.y / max(1, ctx.map_h),
        ctx.energy / max(1, ctx.max_energy),
        ctx.health / max(1, ctx.max_health),
        ctx.nutrinium / max(1, ctx.max_cargo),
        min(ctx.credits, ctx.max_credits) / max(1, ctx.max_credits),
        1.0 if ctx.recharging else 0.0,
        1.0 if ctx.shields_up else 0.0,
    ]

    # Nearest asteroid (4 values)
    nearest_ast = _nearest_entity(ctx.x, ctx.y, ctx.asteroids)
    if nearest_ast:
        dist = _distance(ctx.x, ctx.y, nearest_ast["x"], nearest_ast["y"])
        o.append(nearest_ast["x"] / max(1, ctx.map_w))
        o.append(nearest_ast["y"] / max(1, ctx.map_h))
        o.append(nearest_ast.get("nutrinium", 0) / max(1, nearest_ast.get("mass", 1)))
        o.append(dist / map_span)
    else:
        o.extend([0.0, 0.0, 0.0, 1.0])

    # Nearest trading post (3 values)
    nearest_post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
    if nearest_post:
        dist = _distance(ctx.x, ctx.y, nearest_post["x"], nearest_post["y"])
        o.append(nearest_post["x"] / max(1, ctx.map_w))
        o.append(nearest_post["y"] / max(1, ctx.map_h))
        o.append(dist / map_span)
    else:
o.extend([0.0, 0.0, 1.0])

# First visible (non-destroyed) enemy (5 values)
enemy = ctx.ships[0] if ctx.ships else None
if enemy:
    dist = distance(ctx.x, ctx.y, enemy["x"], enemy["y"])
    o.append(enemy["x"] / max(1, ctx.map_w))
    o.append(enemy["y"] / max(1, ctx.map_h))
    o.append(enemy["health"] / max(1, ctx.max_health))
    o.append(enemy["nutrinium"] / max(1, ctx.max_cargo))
    o.append(dist / map_span)
else:
    o.extend([0.0, 0.0, 0.0, 0.0, 1.0])

return np.array(o, dtype=np.float32)

def _build_sensor_only_observation(ctx):
    """
    Reconstruct the sensor-only observation (mirrors ``SensorOnlyObservationGenerator``).
    
    Layout: ship essentials (6) + a centered, unclamped ``(2*sensor_range+1)^2``
    local grid (enemy 0.9 > trading post 0.7 > asteroid 0.5, else 0; out-of-map
    cells stay 0).
    """
    o = [
        ctx.x / max(1, ctx.map_w),
        ctx.y / max(1, ctx.map_h),
        ctx.energy / max(1, ctx.max_energy),
        ctx.health / max(1, ctx.max_health),
        ctx.nutrinium / max(1, ctx.max_cargo),
        min(ctx.credits, ctx.max_credits) / max(1, ctx.max_credits),
    ]

    sr = ctx.sensor_range
    side = 2 * sr + 1
    grid = [0.0] * (side * side)
    for i in range(-sr, sr + 1):
        for j in range(-sr, sr + 1):
            gx, gy = ctx.x + i, ctx.y + j
            if 0 <= gx < ctx.map_w and 0 <= gy < ctx.map_h:
                idx = (i + sr) * side + (j + sr)
                if _entity_at(gx, gy, ctx.ships):
                    grid[idx] = 0.9
                elif _entity_at(gx, gy, ctx.trading_posts):
                    grid[idx] = 0.7
                elif _entity_at(gx, gy, ctx.asteroids):
                    grid[idx] = 0.5
    o.extend(grid)
    return np.array(o, dtype=np.float32)

# ---------------------------------
# Action masking (delegated to utils.action_masker -- single source of truth)
# ---------------------------------
def _mask_state(ctx):
    """
    Adapt a parsed request context into a `utils.action_masker.MaskState`.
    
    Returns ``None`` if the masking utility cannot be imported (the caller then
    proceeds without masking). The mask is built over SENSOR-VISIBLE entities
    only -- the lambda only sees what the server reports within sensor range.
    """
    masker = _get_masker()
    if masker is None:
        return None
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
        shield_state=ctx._shield_state,
        shield_value=ctx.shield_value,
        shield_capacity=ctx.shield_capacity,
        shields_up=ctx.shields_up,
        modules=ctx.modules,
        negotiate_post_id=ctx.negotiate_post_id,
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
)

# ---------------------------------
# Model prediction + action-mask enforcement
# ---------------------------------

def _predict(model, is_md, obs, model_mask, env_mask, st, spec):
    """Run the model and return `(action_type, target_x, target_y, energy_bin)`.

    The model receives `model_mask` (sized to its own action space). Its raw predicted
    action is mapped back to an env action id via the spec's `ActionSpec` (identity for the 19-action specs) and then run through the shared `action_masker` utility with the full 19-action `env_mask` so the bot only ever commits to a state-valid action.
    """
    obs_dict = {"observation": obs, "action_mask": model_mask}
    vec, _ = model.predict(obs_dict, deterministic=True)

    tx = ty = ebin = None
    arr = np.asarray(vec).reshape(-1)
    if is_md:
        raw_action = int(arr[0]) if arr.size >= 1 else None
        if arr.size >= 3:
            tx, ty = int(arr[1]), int(arr[2])
        if arr.size >= 4:
            ebin = int(arr[3])
    else:
        raw_action = int(arr[0]) if arr.size >= 1 else None

    action_type = None
    if raw_action is not None:
        # Map the model's action id onto an env action id (identity by default).
        if spec is not None:
            try:
                action_type = spec.action_spec.map_action(raw_action, _NUM_ACTIONS)
            except Exception:
                action_type = raw_action
        else:
            action_type = raw_action

    if action_type is not None and st is not None:
        masker = _get_masker()
        if masker is not None:
            action_type = masker.mask_action(action_type, st, mask=env_mask)

    return action_type, tx, ty, ebin


def energy_from_bin(ebin, max_energy):
    """Map a MultiDiscrete energy-bin index to an energy amount (None for bin 0)."""
    if ebin is None or ebin <= 0:
        return None
    frac = min(1.0, ebin / float(max(1, _ENERGY_BINS - 1)))
    return int(round(frac * max_energy))


# ---------------------------------
# Action -> lambda response (inverse of env_translate_bot_action)
# ---------------------------------

def action_to_response(ctx, action_type, tx, ty, ebin):
    """Convert a chosen env action id into a lambda response dict."""
    if action_type is None or not (0 <= action_type < _NUM_ACTIONS):
        return {"actionType": "WAIT"}
    name = _ACTION_NAMES[action_type]

    if name in _SIMPLE_ACTIONS:
        return {"actionType": name}

    if name == "MOVE_NORTH":
        return {"actionType": "MOVE", "payload": {"direction": "N"}}
    if name == "MOVE_SOUTH":
        return {"actionType": "MOVE", "payload": {"direction": "S"}}
    if name == "MOVE_EAST":
        return {"actionType": "MOVE", "payload": {"direction": "E"}}
    if name == "MOVE_WEST":
        return {"actionType": "MOVE", "payload": {"direction": "W"}}

    if name == "SELL":
        return {"actionType": "SELL", "payload": {"nutrinium": int(ctx.nutrinium)}}

    if name == "ATTACK":
        enemy = _entity_at(ctx.x, ctx.y, ctx.ships)
        if enemy is None:
            return {"actionType": "WAIT"}
        energy = _energy_from_bin(ebin, ctx.max_energy)
        if not energy:
            energy = _DEFAULT_ATTACK_ENERGY
        return {
            "actionType": "ATTACK",
            "payload": {"target": enemy.get("playerId"), "energy": int(energy)},
        }

    if name == "JUMP_TO_ASTEROID":
        target = None
        if tx is not None and ty is not None and (tx, ty) != (ctx.x, ctx.y):
            target = {"x": tx, "y": ty}
        else:
            top = _top_asteroids(ctx, ctx.x, ctx.y, 1)
if top:
    target = {"x": top[0]["x"], "y": top[0]["y"]}
if target is None:
    return {"actionType": "WAIT"}
return {"actionType": "JUMP", "payload": {"target_location": target}}

if name == "JUMP_TO_TRADING_POST":
    post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
if post is None:
    return {"actionType": "WAIT"}
return {
    "actionType": "JUMP",
    "payload": {"target_location": {"x": post["x"], "y": post["y"]}},
}

return {"actionType": "WAIT"}

# --------------------------------
# Public lambda contract
# --------------------------------
def get_model_action(action_request):
    """Decide an action for one ActionRequest via the trained model (raw dict)."""
    if not _NUMPY_OK:
        return {"actionType": "WAIT"}

    loaded = _load_model()
    if loaded is None:
        return {"actionType": "WAIT"}
    model, is_md = loaded

    spec = _load_spec()
    ctx = _Context(action_request)
    obs = _build_observation(ctx, spec)

    # The builder now appends the 38-value action-restriction block (total 262 for
    # the full layout). Models trained before that block expect the shorter vector,
    # so truncate the reconstructed observation to the loaded model's own size. A
    # future model retrained on the extended layout consumes the full vector.
    try:
        model_obs_space = model.observation_space
        model_obs_size = None
        if hasattr(model_obs_space, "spaces") and "observation" in model_obs_space.spaces:
            model_obs_size = int(model_obs_space.spaces["observation"].shape[0])
        elif hasattr(model_obs_space, "shape") and model_obs_space.shape:
            model_obs_size = int(model_obs_space.shape[0])
        if model_obs_size is not None and obs.shape[0] > model_obs_size:
            obs = obs[:model_obs_size]
    except Exception:
        pass

    # Validate the observation length when the spec declares an explicit size;
    # otherwise trust the builder (the model.predict call enforces the rest).
    expected_size = None
    num_actions = _NUM_ACTIONS
    if spec is not None:
        try:
            expected_size = spec.observation_spec.observation_size
        except Exception:
            expected_size = None
        try:
            num_actions = int(spec.action_spec.action_space_size)
        except Exception:
            num_actions = _NUM_ACTIONS
    if expected_size is not None and obs.shape[0] != expected_size:
        return {"actionType": "WAIT"}

    st = _mask_state(ctx)
    masker = _get_masker()
    if masker is not None and st is not None:
        env_mask = masker.get_action_mask(st)
    else:
        env_mask = np.ones(_NUM_ACTIONS, dtype=np.int8)

    # The mask handed to the model must match its own action-space size.
    if num_actions == _NUM_ACTIONS:
        model_mask = env_mask
    elif num_actions < _NUM_ACTIONS:
        model_mask = env_mask[:num_actions]
    else:
        model_mask = np.ones(num_actions, dtype=np.int8)
        model_mask[:_NUM_ACTIONS] = env_mask

    action_type, tx, ty, ebin = _predict(model, is_md, obs, model_mask, env_mask, st, spec)
    return action_to_response(ctx, action_type, tx, ty, ebin)

def get_action(action_request):
    """Public entry point: returns a normalised response dict."""
    try:
        return to_response(get_model_action(action_request))
    except Exception:
        return {"actionType": "WAIT"}