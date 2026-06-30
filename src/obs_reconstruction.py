"""Shared observation + action-mask reconstruction from an ActionRequest.

Single source of truth used by BOTH:

  * the model-backed bot (:mod:`bots.bot_v6`) at inference time, and
  * the environment (:class:`ProspectorsPiratesEnv`) when
    ``partial_observability`` training is enabled,

so a model trains on byte-identical observations/masks to what it sees as a
delegating bot. Because both sides run the same builder over the same
sensor-limited ``ActionRequest`` (produced by
``EnvOpponentMixin._compose_action_request``), parity is exact by construction.

The reconstruction is intentionally *myopic*: it only consumes what the server
reports in ``sensors`` (entities within sensor range), mirroring the live-lambda
contract. The global, full-map observation builder lives in
``env_observation_mixin`` and is used only when partial observability is OFF.
"""

import math
import os
import sys
import time

try:
    import numpy as np
    _NUMPY_OK = True
except Exception:  # pragma: no cover - numpy is expected but degrade gracefully
    np = None
    _NUMPY_OK = False


# ----------------------------------------------------------------------------
# Constants (shared by the env training path and the bot inference path)
# ----------------------------------------------------------------------------
_NUM_ACTIONS = 19
_ENERGY_BINS = 11               # MultiDiscrete energy dimension (matches the env)

# Environment config constants that the observation normalizers depend on but
# that the ActionRequest does not carry (they match the simulator's config).
_MAX_CARGO = 1000               # max_nutrinium_cargo
_MAX_CREDITS = 10000            # max_credits
_MAX_HEALTH = 100               # max_health
_MAX_SKILL_POINTS = 24          # max_skill_points
_TOP_ASTEROIDS_COUNT = 5        # top_asteroids_count
_MAX_STEPS = 300                # episode length proxy for the action counter
_MARKET_REF = 98.0              # config market.sell_nutrinium reference price
_SPATIAL_REF = 50.0             # fixed reference length for scale-free deltas/distances


def _scaled_delta(d):
    """Scale a signed coordinate delta to [-1, 1] by a fixed reference length.

    Map-size invariant; mirrors env_observation_mixin._scaled_delta exactly so
    the reconstructed observation matches the env for the same world geometry.
    """
    return max(-1.0, min(1.0, d / _SPATIAL_REF))


def _scaled_distance(dist):
    """Scale a non-negative distance to [0, 1) via dist / (dist + ref).

    Map-size invariant; mirrors env_observation_mixin._scaled_distance exactly.
    """
    return dist / (dist + _SPATIAL_REF)


# Action id -> name (matches env_common.ActionType, display/translate order).
_ACTION_NAMES = [
    "WAIT", "MINE", "MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST",
    "RECHARGE", "RECHARGE_END", "ATTACK", "JUMP_TO_ASTEROID", "SELL",
    "RAISE_SHIELDS", "JUMP_TO_TRADING_POST", "RESPAWN", "PLUNDER", "SALVAGE",
    "REPAIR", "NEGOTIATE", "LOWER_SHIELDS",
]


def _restriction_key(action_name):
    """Map an action name to its ``metadata.actionRestrictions`` key.

    The four MOVE directions share the ``MOVE`` rule and both JUMP variants share
    the ``JUMP`` rule; every other action maps to itself. Mirrors
    ``utils.action_masker.ACTION_RESTRICTION_NAME``.
    """
    if action_name.startswith("MOVE_"):
        return "MOVE"
    if action_name.startswith("JUMP_"):
        return "JUMP"
    return action_name


# Action id -> restriction key, aligned to the 19-action mask order.
_ACTION_RESTRICTION_KEY = [_restriction_key(n) for n in _ACTION_NAMES]


# ----------------------------------------------------------------------------
# Lazy action_masker import (cached at module scope)
# ----------------------------------------------------------------------------
_MASKER = "unset"  # sentinel -> the utils.action_masker module, or None on failure


def _get_masker():
    """Lazily import and cache ``utils.action_masker``. Returns the module or None.

    The ``src`` directory (this module's own directory) is added to ``sys.path``
    so the utility resolves both inside the simulator and when run as a
    standalone lambda. Any failure degrades gracefully (callers fall back to no
    masking).
    """
    global _MASKER
    if _MASKER != "unset":
        return _MASKER
    result = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # .../src
        if here not in sys.path:
            sys.path.insert(0, here)
        from utils import action_masker
        result = action_masker
    except Exception:
        result = None
    _MASKER = result
    return result


# ----------------------------------------------------------------------------
# Geometry / parsing helpers (stateless)
# ----------------------------------------------------------------------------
def _distance(x1, y1, x2, y2):
    """Euclidean distance between two cells."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


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


def _ab(skills, name, default):
    """Skill/ability value (numeric), falling back to ``default``."""
    try:
        v = skills.get(name, default)
        return float(v if v is not None else default)
    except (AttributeError, TypeError, ValueError):
        return float(default)


# ----------------------------------------------------------------------------
# Parsed request
# ----------------------------------------------------------------------------
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
        self.shield_state = str(shield.get("state", "DOWN") or "DOWN").upper()
        self.shield_value = float(shield.get("value", 0) or 0)
        self.shield_capacity = float(shield.get("capacity", 0) or 0)
        self.shields_up = self.shield_state == "POWERED"

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
                entry["health"] = int(s.get("health", _MAX_HEALTH))
                entry["energy"] = int(s.get("energy", 0))
                entry["credits"] = int(s.get("credits", 0))
                entry["nutrinium"] = int(s.get("nutrinium", 0))
                entry["playerId"] = s.get("playerId")
                entry["teamId"] = s.get("teamId")
                entry["skills"] = s.get("skills", {}) or {}
                sh = s.get("shield", {}) or {}
                entry["shield_state"] = str(sh.get("state", "") or "").upper()
                self.ships.append(entry)

        # Round metadata: economics + map size + market price.
        game_state = req.get("gameState", {}) or {}
        self.game_id = game_state.get("gameId")
        self.game_round = game_state.get("round", 0)
        self.tick = int(game_state.get("tick", 0) or 0)
        # Wall-clock game bounds (Unix-ms). Present in live requests, absent in the
        # env-composed request -> remaining time falls back to the tick estimate.
        self.game_start = game_state.get("start")
        self.game_end = game_state.get("end")
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
        self.max_jump_distance = int(ship_cfg.get("maxJumpDistance", 50))
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
        self.max_health = _MAX_HEALTH
        self.max_cargo = _MAX_CARGO
        self.max_credits = _MAX_CREDITS
        self.max_skill_points = _MAX_SKILL_POINTS
        self.top_count = _TOP_ASTEROIDS_COUNT
        self.max_steps = _MAX_STEPS

    def jump_energy_cost(self, distance):
        """Energy cost of a jump of the given distance (skill lowers the floor)."""
        adj_min = max(0, self.jump_min_cost - int(_ab(self.skills, "jump_cost", 0)) * 5)
        return int(max(adj_min, round(self.jump_unit_cost * distance)))


# Public alias.
Context = _Context


# ----------------------------------------------------------------------------
# Observation reconstruction helpers (mirror env_observation_mixin)
# ----------------------------------------------------------------------------
def _top_asteroids(ctx, x, y, count):
    """Top-N asteroids by the env's score (concentration * nutrinium / dist).

    Mirrors ``EnvObservationMixin._get_top_asteroids`` but over sensor-visible
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
    return scored[:count]


def _combat_score(enemy, raw=False):
    """Weighted enemy combat score (mirrors ``_calculate_enemy_combat_score``)."""
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


def _extreme_enemies(ctx):
    """Strongest / weakest visible enemy (mirrors ``_get_extreme_enemies``)."""
    active = list(ctx.ships)
    if not active:
        return None, None
    if len(active) == 1:
        return active[0], active[0]
    scored = sorted(active, key=lambda e: _combat_score(e, raw=True), reverse=True)
    return scored[0], scored[-1]


def _prey_enemies(ctx, count=3):
    """Top N weakest huntable enemies (mirrors ``_get_prey_enemies``).

    Over the sensor-visible contacts (``ctx.ships``), keep enemies that are not
    teammates, hold nutrinium, and are weaker than the player in BOTH attack
    (attack_power + attack_accuracy) AND defense (shield_strength + evade).
    Ranked weakest-first by raw combat score. Each: ``{'x', 'y', 'nutrinium'}``.
    """
    player_attack = _ab(ctx.skills, "attack_power", 0) + _ab(ctx.skills, "attack_accuracy", 0)
    player_defense = _ab(ctx.skills, "shield_strength", 0) + _ab(ctx.skills, "evade", 0)

    candidates = []
    for enemy in ctx.ships:
        if enemy.get("nutrinium", 0) <= 0:
            continue
        enemy_team = enemy.get("teamId")
        if enemy_team is not None and int(enemy_team) == ctx.team_id:
            continue
        eskills = enemy.get("skills", {}) or {}
        enemy_attack = _ab(eskills, "attack_power", 0) + _ab(eskills, "attack_accuracy", 0)
        enemy_defense = _ab(eskills, "shield_strength", 0) + _ab(eskills, "evade", 0)
        if not (enemy_attack < player_attack and enemy_defense < player_defense):
            continue
        candidates.append((_combat_score(enemy, raw=True), enemy))

    candidates.sort(key=lambda c: c[0])
    return [
        {"x": e["x"], "y": e["y"], "nutrinium": e.get("nutrinium", 0)}
        for _, e in candidates[:count]
    ]


def _remaining_time_fraction(ctx):
    """Fraction of the game still remaining, in [0, 1] (1.0 at start -> 0.0 at end).

    Live requests carry wall-clock bounds (``gameState.start`` / ``end`` in Unix-ms),
    so inference reads the true time left. The env-composed training request has
    neither, so it falls back to ``(max_steps - tick) / max_steps`` -- byte-identical
    to the env FULL builder's action-counter normalization (parity under
    partial-observability training).
    """
    start = ctx.game_start
    end = ctx.game_end
    if start is not None and end is not None:
        try:
            span = float(end) - float(start)
            if span > 0:
                now_ms = time.time() * 1000.0
                return max(0.0, min(1.0, (float(end) - now_ms) / span))
        except (TypeError, ValueError):
            pass
    return max(0.0, min(1.0, (ctx.max_steps - ctx.tick) / max(1, ctx.max_steps)))


def _quadrant_norm(x, y, map_w, map_h):
    """Player's cell in a 3x3 map grid as a single normalized index q/8 (q in 0..8)."""
    col = min(2, (int(x) * 3) // max(1, int(map_w)))
    row = min(2, (int(y) * 3) // max(1, int(map_h)))
    return (row * 3 + col) / 8.0


def _build_observation(ctx, spec=None):
    """Dispatch to the observation builder matching ``spec``'s observation type.

    Supports ``full`` (275-dim), ``full_no_grid`` (154-dim), ``compact`` (57-dim)
    and ``sensor_only`` (6 + (2*sensor_range+1)^2). Anything else (or a missing
    spec) builds the full observation.
    """
    obs_type = "full"
    if spec is not None:
        try:
            obs_type = spec.observation_spec.observation_type
        except Exception:
            obs_type = "full"
    if obs_type == "full_no_grid":
        return _build_full_observation(ctx, include_sensor_grid=False)
    if obs_type == "compact":
        return _build_compact_observation(ctx)
    if obs_type == "sensor_only":
        return _build_sensor_only_observation(ctx)
    return _build_full_observation(ctx)


def _build_full_observation(ctx, include_sensor_grid=True):
    """Reconstruct the env's full observation vector from the request.

    With ``include_sensor_grid`` True (default) this yields the 275-dim FULL
    layout; with it False it yields the 154-dim FULL_NO_GRID layout (the local
    sensor-grid block is omitted), byte-identical to the env's
    ``_get_observation(include_sensor_grid=False)``.
    """
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
    o.append(ctx.skill_points_spent / max(1, ctx.max_skill_points))
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
        o.append(_scaled_delta(top1[0]["x"] - ctx.x))
        o.append(_scaled_delta(top1[0]["y"] - ctx.y))
    else:
        o.extend([0.0, 0.0])
    nearest_tp = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
    if nearest_tp:
        o.append(_scaled_delta(nearest_tp["x"] - ctx.x))
        o.append(_scaled_delta(nearest_tp["y"] - ctx.y))
    else:
        o.extend([0.0, 0.0])

    # === LOCAL SENSOR GRID (11x11 = 121 values) ===
    # Omitted entirely for the FULL_NO_GRID layout (include_sensor_grid=False).
    if include_sensor_grid:
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
    max_mass = max(1.0, ctx.asteroid_mass_max)
    for a in top5:
        o.append(a["x"] / max(1, ctx.map_w))
        o.append(a["y"] / max(1, ctx.map_h))
        o.append(a["mass"] / max_mass)
        o.append(a["nutrinium"] / max_mass)
        o.append(_scaled_distance(a["distance"]))
        o.append(a["score"])
    for _ in range(ctx.top_count - len(top5)):
        o.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # === NEAREST TRADING POST (3 values) ===
    if nearest_tp:
        dist = _distance(ctx.x, ctx.y, nearest_tp["x"], nearest_tp["y"])
        o.append(nearest_tp["x"] / max(1, ctx.map_w))
        o.append(nearest_tp["y"] / max(1, ctx.map_h))
        o.append(_scaled_distance(dist))
    else:
        o.extend([0.0, 0.0, 0.0])

    # === TWO ENEMY TYPES (16 values) ===
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
            enemy_team = enemy.get("teamId")
            o.append(1.0 if (enemy_team is not None and int(enemy_team) == ctx.team_id) else 0.0)
        else:
            o.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # === SPEC-FIDELITY FEATURES (24 values) ===
    o.append(_ab(sk, "shield_capacity", 0) / 10.0)
    o.append(_ab(sk, "shield_efficiency", 0) / 10.0)
    o.append(_ab(sk, "jump_cost", 0) / 10.0)
    o.append(_ab(sk, "salvage_yield", 0) / 10.0)
    o.append(_ab(sk, "negotiate_skill", 0) / 10.0)
    o.append(_ab(sk, "negotiate_caution", 0) / 10.0)
    o.append(_ab(sk, "negotiate_ambition", 0) / 10.0)

    scap = ctx.shield_capacity
    sval = ctx.shield_value
    max_capacity = ctx.base_shield_capacity + 10 * 10  # base + max_abil shield_capacity*10
    o.append(1.0 if ctx.shield_state == "POWERED" else 0.0)
    o.append(1.0 if ctx.shield_state == "DRAINING" else 0.0)
    o.append(1.0 if ctx.shield_state == "DOWN" else 0.0)
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
        o.append(_scaled_delta(obj_post["x"] - ctx.x))
        o.append(_scaled_delta(obj_post["y"] - ctx.y))
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
        o.append(_scaled_delta(nearest_wreck["x"] - ctx.x))
        o.append(_scaled_delta(nearest_wreck["y"] - ctx.y))
    else:
        o.extend([0.0, 0.0, 0.0])

    # Action restrictions (38 values: 19 actions * 2 flags), mirroring the env's
    # _action_restriction_features. Encodes the request's actionRestrictions matrix
    # so a restriction-aware model can adapt; missing rules default to allowed.
    restrictions = ctx.action_restrictions or {}
    for key in _ACTION_RESTRICTION_KEY:
        rule = restrictions.get(key, {}) if key is not None else {}
        o.append(1.0 if rule.get("allowedWhileRecharging", True) else 0.0)
        o.append(1.0 if rule.get("allowedWithShieldsUp", True) else 0.0)

    # === TEMPORAL/SPATIAL (2 values, appended last) ===
    # remaining_time_fraction: fraction of the game still left (1.0 -> 0.0). A live
    #   request carries wall-clock bounds (start/end, Unix-ms) so the bot reads true
    #   time remaining; the env-composed request has neither, so training falls back
    #   to the tick estimate -- byte-identical to the env FULL builder.
    # quadrant_norm: player's cell in a 3x3 map grid as a single normalized index.
    o.append(_remaining_time_fraction(ctx))
    o.append(_quadrant_norm(ctx.x, ctx.y, ctx.map_w, ctx.map_h))

    # === PREY ENEMIES (9 values: 3 weakest huntable enemies * 3 features) ===
    # Appended after temporal/spatial so legacy offsets stay stable. Top 3 weakest
    # non-teammate enemies with weaker attack AND defense than the player, holding
    # nutrinium, among the sensor-visible contacts. Each: (x/W, y/H, nutrinium).
    prey = _prey_enemies(ctx, count=3)
    for p in prey:
        o.append(p["x"] / max(1, ctx.map_w))
        o.append(p["y"] / max(1, ctx.map_h))
        o.append(min(p.get("nutrinium", 0), 100) / 100.0)
    for _ in range(3 - len(prey)):
        o.extend([0.0, 0.0, 0.0])

    return np.array(o, dtype=np.float32)


def _build_compact_observation(ctx):
    """Reconstruct the 57-dim compact observation (mirrors ``CompactObservationGenerator``).

    Layout: ship state (8) + top 5 asteroids (30) + nearest trading post (3) +
    two enemy types (16), over sensor-visible entities only.
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

    # Top 5 asteroids (30 values: 5 asteroids * 6 features)
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
        o.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Nearest trading post (3 values)
    nearest_post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
    if nearest_post:
        dist = _distance(ctx.x, ctx.y, nearest_post["x"], nearest_post["y"])
        o.append(nearest_post["x"] / max(1, ctx.map_w))
        o.append(nearest_post["y"] / max(1, ctx.map_h))
        o.append(dist / map_span)
    else:
        o.extend([0.0, 0.0, 1.0])

    # Two enemy types (16 values: 2 enemies * 8 features)
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
            enemy_team = enemy.get("teamId")
            o.append(1.0 if (enemy_team is not None and int(enemy_team) == ctx.team_id) else 0.0)
        else:
            o.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return np.array(o, dtype=np.float32)


def _build_sensor_only_observation(ctx):
    """Reconstruct the sensor-only observation (mirrors ``SensorOnlyObservationGenerator``).

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


# ----------------------------------------------------------------------------
# Action masking (delegated to utils.action_masker -- single source of truth)
# ----------------------------------------------------------------------------
def _mask_state(ctx):
    """Adapt a parsed request context into a ``utils.action_masker.MaskState``.

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
        shield_state=ctx.shield_state,
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
        jump_min_cost=ctx.jump_min_cost,
        jump_cost_skill=int(_ab(ctx.skills, "jump_cost", 0)),
        max_jump_distance=ctx.max_jump_distance + int(_ab(ctx.skills, "jump_distance", 0)) * 10,
    )


# ----------------------------------------------------------------------------
# Public API (used by both the bot and the environment)
# ----------------------------------------------------------------------------
def build_observation(action_request, spec=None):
    """Build the reconstructed observation vector from an ``ActionRequest``.

    Args:
        action_request: The server/simulator ActionRequest dict.
        spec: Optional ``model_specs.ModelSpec`` selecting the observation type
            (``full`` / ``compact`` / ``sensor_only``). Defaults to full.

    Returns:
        ``np.ndarray`` (float32) observation vector.
    """
    ctx = _Context(action_request)
    return _build_observation(ctx, spec)


def build_action_mask(action_request):
    """Build the 19-action validity mask from an ``ActionRequest``.

    Mirrors the env's per-state mask but over SENSOR-VISIBLE entities only, so a
    training env using this matches what the bot computes at inference. Falls
    back to an all-ones mask if ``utils.action_masker`` cannot be imported.

    Returns:
        ``np.ndarray`` (int8) of length 19.
    """
    ctx = _Context(action_request)
    st = _mask_state(ctx)
    masker = _get_masker()
    if masker is not None and st is not None:
        return masker.get_action_mask(st)
    return np.ones(_NUM_ACTIONS, dtype=np.int8)
