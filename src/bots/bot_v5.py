"""
Prospectors & Pirates bot (v5).
A standalone strategy port of the 2025 `r680329` competition bot (`spaceship.SpaceShip` + `action.Action`) onto the 2026 lambda contract shared by `bot_v2` / `bot_v3` / `bot_v4`: a single :func:`get_action` that takes an ActionRequest dict and returns `{"actionType": str, "payload": ...}`.

The 2025 bot is a BALANCED MINER-TRADER with opportunistic raiding and -- its signature -- market-timing on sells (bank cargo when the price is high relative to the round's running low/high). This port keeps that personality while adapting to the 2026 game model:

* The 2025 action-point budget is gone (2026 is one action per tick) -- all action-point bookkeeping is dropped.
* The 2025 COMBAT sub-system (allocating energy to WEAPONS/SHIELDS/THRUSTERS) is collapsed onto the 2026 single-shot `ATTACK(target, energy)`, sizing the energy from the same energy-matchup prey-selection rules.
* The 2025 dense `long` sensor matrix is replaced by the 2026 flat `sensors` list (entities within sensor range); the zone-valuation formula is applied per visible asteroid.
* 2026-only mobility (`JUMP`) is used to reach far targets, like bot_v3/v4.

Decision priority (mirrors `generate_action_response`):

1. Respawn if destroyed.
2. Energy management: recharge when low, end recharge once topped up.
3. Threat-aware banking: carrying cargo + a stronger ship nearby -> sell on a post, else flee toward the nearest post.
4. Market-timing sell: when the price/round-history says it's time, sell on a post (navigate to one otherwise).
5. Opportunistic attack: overpower a profitable, beatable ship in our cell.
6. Mine-vs-move: mine the asteroid under us when it's our target zone or rich enough, otherwise travel toward the best zone (jump when far + affordable).

Directions are emitted in the live server frame (N=y+1, S=y-1, E=x+1, W=x-1), matching the environment's MOVE actions, so no axis calibration is needed.

NOTE: like the other bots, this one only sees what the server reports in `sensors` (entities within sensor range), so its choices are necessarily local compared to the in-environment AIs that have global map knowledge.
"""

import os
import random
import sys

def _to_response(payload):
    """
    Lightweight lambda response adapter (mirrors bot_v2/bot_v3/bot_v4).
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


# Tunables (ported from the 2025 bot, adapted to the 2026 energy economy)
RECHARGE_LOW_FRAC = 0.10      # recharge when energy drops below 10% of max
RECHARGE_END_FRAC = 0.60     # stop recharging once back to ~60% of max
NUTRINIUM_MIN_RATIO = 0.7   # mine in place when concentration (nutr/mass) > 0.7
MIN_SELL_QUANTITY = 20       # don't bother selling below this cargo (keep mining)
HARD_SELL_QUANTITY = 50      # always sell once cargo reaches this
HAUL_CARGO_THRESHOLD = 12    # cargo worth carrying toward a post when idle
MAX_MINING_SEARCH_TICK = 10  # ignore candidate zones beyond this Manhattan range
DEFAULT_ATTACK_ENERGY = 20   # baseline ATTACK payload
ATTACK_CRITICAL_MARGIN = 5  # extra energy committed to ensure an overpower


# Optional id hooks (extensible; empty by default). The 2025 bot kept hardcoded "friend"/"attacker" player-id lists; they are irrelevant to the 2026 simulator and lambda, so threat/prey decisions here are purely energy-based.
_FRIEND_IDS = set()
_ATTACKER_IDS = set()


# Module-level market history (the 2025 buy-low / sell-high memory).
# Keyed on gameId so interleaved opponents in the simulator don't clobber each other; reset when the round changes (matches the 2025 per-round reset).
_market_high = {}  # gameId -> highest sell price seen this round
_market_low = {}   # gameId -> lowest sell price seen this round
_market_round = {}  # gameId -> round the history above belongs to

def reset_market_state():
    """Clear the market-history globals (test isolation helper)."""
    _market_high.clear()
    _market_low.clear()
    _market_round.clear()

def update_market_history(game_id, game_round, price):
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

# ----------------------------------------------
# Geometry / parsing helpers (stateless)
# ----------------------------------------------

def distance(x1, y1, x2, y2):
    """Euclidean distance between two cells."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def manhattan(x1, y1, x2, y2):
    """Manhattan distance between two cells (zone-valuation tick proxy)."""
    return abs(x1 - x2) + abs(y1 - y2)

def skill(skills, name):
    """Skill/ability level (0 when absent)."""
    try:
        return int(skills.get(name, 0) or 0)
    except (AttributeError, TypeError, ValueError):
        return 0

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
        d = distance(x, y, e["x"], e["y"])
        if best_dist is None or d < best_dist:
            best_dist = d
            best = e
    return best

def direction_towards(dx, dy):
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

# Parsed request
# --------------------------------------------------------

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

        # Split sensor contacts by type into flat {x, y, ...} dicts.
        self.asteroids = []
        self.trading_posts = []
        self.ships = []
        for s in req.get("sensors", []):
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
                self.ships.append(entry)

        # Total visible mineable nutrinium -> MINING vs ATTACKING mode proxy.
        self.total_remaining = sum(a["nutrinium"] for a in self.asteroids)

    # Round metadata: economics + map size + market price.
    game_state = req.get("gameState", {}) or {}
    self.game_id = game_state.get("gameId")
    self.game_round = game_state.get("round", 0)
    self.tick = int(game_state.get("tick", 0) or 0)
    metadata = game_state.get("metadata", {} or {})
    ship_cfg = metadata.get("shipConfig", {} or {})
    costs = ship_cfg.get("energyCosts", {} or {})
    self.mine_cost = int(costs.get("mine", 10))
    self.attack_cost = int(costs.get("attack", 1))
    self.jump_unit_cost = int(costs.get("jump", 1))
    self.jump_min_cost = int(costs.get("jumpMinCost", 75))
    self.max_energy = int(ship_cfg.get("maxEnergy", 100))
    self.per_recharge = int(ship_cfg.get("energyPerRecharge", 10))
    # Extra fields the shared action-masker safety net validates against.
    self.move_cost = int(costs.get("move", 1))
    self.shields_cost = int(costs.get("shields", 5))
    self.plunder_cost = int(costs.get("plunder", 5))
    self.negotiate_cost = int(costs.get("negotiate", 5))
    self.max_health = int(ship_cfg.get("maxHealth", 100))
    self.credits = int(me.get("credits", 0))
    map_cfg = metadata.get("mapConfig", {} or {})
    self.map_w = int(map_cfg.get("width", 10))
    self.map_h = int(map_cfg.get("height", 10))
    market = metadata.get("market", {} or {})
    sell_cfg = market.get("sell", {} or {})
    price = sell_cfg.get("nutrinium")
    self.market_price = float(price) if price is not None else 0.0
    # Per-state action restrictions (allowedWhileRecharging / allowedWithShieldsUp).
    self.action_restrictions = metadata.get("actionRestrictions", {} or {})

def jump_energy_cost(self, distance):
    """Energy cost of a jump of the given distance (skill lowers the floor)."""
    adj_min_cost = max(0, self.jump_min_cost - _skill(self.skills, "jump_cost") * 5)
    return int(max(adj_min_cost, round(self.jump_unit_cost * distance)))

# Action builders
# --------------------------------------------------------

def _move(direction):
    return {"actionType": "MOVE", "payload": {"direction": direction}}

def jump(target):
    return {"actionType": "JUMP",
            "payload": {"target_location": {"x": target["x"], "y": target["y"]}}}
def _sell(amount):
    return {"actionType": "SELL", "payload": {"nutrinium": int(amount)}}

def _attack(target, energy):
    return {"actionType": "ATTACK",
            "payload": {"target": target.get("playerId"), "energy": int(energy)}}

def _navigate(ctx, tx, ty, jump_min_dist, jump_margin):
    """Travel toward (tx, ty): jump when far and affordable, else step."""
    dist = _distance(ctx.x, ctx.y, tx, ty)
    if dist > jump_min_dist and ctx.energy >= ctx.jump_energy_cost(dist) + jump_margin:
        return _jump({"x": tx, "y": ty})
    return _move(_direction_towards(tx - ctx.x, ty - ctx.y))

def _move_explore(ctx):
    """Step in a random in-bounds direction (2025 '__move_random__')."""
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

# Strategy sub-procedures
def has_potential_attacker(ctx):
    """True when a visible ship clearly outguns us (2025 '__has_potential_attacker__')."""
    if any(s.get("playerId") in _ATTACKER_IDS for s in ctx.ships):
        return True
    strongest = None
    for s in ctx.ships:
        if strongest is None or s["energy"] > strongest["energy"]:
            strongest = s
    return strongest is not None and ctx.energy * 2 < strongest["energy"]

def find_target_zone(ctx):
    """Best mining zone among visible asteroids by 2025 zone valuation."""
    best = None
    best_value = 0.0
    for ast in ctx.asteroids:
        if ast["nutrinium"] <= 0 or ast["mass"] <= 0:
            continue
        dist = _manhattan(ctx.x, ctx.y, ast["x"], ast["y"])
        if dist > MAX_MINING_SEARCH_TICK:
            continue
        ship_count = len(entities_at(ast["x"], ast["y"], ctx.ships))
        value = _zone_value(ast["nutrinium"], ast["mass"], ship_count, dist)
        if value > best_value:
            best_value = value
            best = {"x": ast["x"], "y": ast["y"], "value": value}
    return best

def find_prey(ctx):
    """Pick a beatable, profitable ship in our cell; size the ATTACK energy.

    Ports the 2025 energy-matchup prey selection: prefer the richest cargo we
    can overpower (their energy <= ours). In MINING mode only interrupt mining
    for a target richer than what the asteroid under us is worth.

    Returns `(prey_or_None, energy_to_commit)`.
    """
    if ctx.energy < ctx.attack_cost:
        return None, 0
    same_zone = entities_at(ctx.x, ctx.y, ctx.ships)
    candidates = [s for s in same_zone
                  if s.get("playerId") != ctx.player_id
                  and s.get("playerId") not in _FRIEND_IDS]
    if not candidates:
        return None, 0

    mining_mode = ctx.total_remaining > 0
    asteroid_here = entity_at(ctx.x, ctx.y, ctx.asteroids)
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
def _balanced_action(ctx):
    # === 1. RESPAWN if destroyed ===
    if ctx.state == "DESTROYED":
        return {"actionType": "RESPAWN"}

    # === 2. ENERGY MANAGEMENT ===
    if ctx.recharging:
        if ctx.energy + ctx.per_recharge > ctx.max_energy:
            return {"actionType": "RECHARGE_END"}
        if ctx.energy >= RECHARGE_END_FRAC * ctx.max_energy:
            return {"actionType": "RECHARGE_END"}
        return {"actionType": "WAIT"}
    if ctx.energy <= 0:
        return {"actionType": "RECHARGE"}
    if ctx.energy < RECHARGE_LOW_FRAC * ctx.max_energy:
        return {"actionType": "RECHARGE"}

    # === 3. THREAT-AWARE BANKING: protect cargo from a stronger neighbour ===
    if ctx.nutrinium > 0 and _has_potential_attacker(ctx):
        if _entity_at(ctx.x, ctx.y, ctx.trading_posts):
            return sell(ctx.nutrinium)
        flee = _go_to_post(ctx)
        if flee is not None:
            return flee

    # === 4. MARKET-TIMING SELL ===
    if _is_selling_time(ctx):
        if _entity_at(ctx.x, ctx.y, ctx.trading_posts):
            return sell(ctx.nutrinium)
        to_post = _go_to_post(ctx)
        if to_post is not None:
            return to_post
        # No post in range; fall through and keep working.

    # === 5. OPPORTUNISTIC ATTACK ===
    prey, commit = _find_prey(ctx)
    if prey is not None:
        return attack(prey, commit)

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
MASKER = "unset"  # sentinel -> the utils.action_masker module, or None on failure

def _get_masker():
    """Lazy import and cache `utils.action_masker`. Returns the module or None.

    The `src` directory (parent of `bots`) is added to `sys.path` so the
    utility resolves both inside the simulator and when run as a standalone
    lambda. Any failure (e.g. numpy missing) degrades gracefully -- the caller
    keeps the heuristic's action unmasked.
    """
    global MASKER
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
    """Adapt the parsed balanced-bot context into a `MaskState`.

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
    )

def action_name_to_id(masker, action, ctx):
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
        return move("N")
    if action_id == masker.MOVE_SOUTH:
        return move("S")
    if action_id == masker.MOVE_EAST:
        return move("E")
    if action_id == masker.MOVE_WEST:
        return move("W")
    if action_id == masker.SELL:
        return sell(ctx.nutrinium)
    if action_id == masker.ATTACK:
        enemy = _richest_zone_enemy(ctx)
        if enemy is None:
            return {"actionType": "WAIT"}
        return attack(enemy, min(DEFAULT_ATTACK_ENERGY, ctx.energy))
    if action_id == masker.JUMP_TO_ASTEROID:
        ast = _nearest_entity(ctx.x, ctx.y, [a for a in ctx.asteroids if a["nutrinium"] > 0])
        return jump(ast) if ast else {"actionType": "WAIT"}
    if action_id == masker.JUMP_TO_TRADING_POST:
        post = _nearest_entity(ctx.x, ctx.y, ctx.trading_posts)
        return jump(post) if post else {"actionType": "WAIT"}
    return {"actionType": "WAIT"}

def enforce(ctx, action):
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

# -----------------------------------
# Public lambda contract
# -----------------------------------
def get_heuristic_action(action_request):
    """Decide a balanced miner-trader action for one ActionRequest (raw dict)."""
    ctx = _Context(action_request)
    _update_market_history(ctx.game_id, ctx.game_round, ctx.market_price)
    return _enforce(ctx, _balanced_action(ctx))

def get_action(action_request):
    """Public entry point: returns a normalised response dict."""
    return _to_response(get_heuristic_action(action_request))