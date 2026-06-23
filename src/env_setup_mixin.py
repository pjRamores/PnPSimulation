"""
Setup / loader mixin for :class:`ProspectorsPiratesEnv`.

Owns config-file loaders (predefined asteroids, trading posts, start position,
enemy-model entries/paths), per-episode ability/module generation, the small
ship-state accessors (:meth:`_skill`, :meth:`_shield_state`,
:meth:`_action_allowed`, :meth:`_has_module`), negotiate-objective assignment
and enemy-model loading/availability bookkeeping.
"""
from env_common import *

# --- Archetype-biased skill distribution ------------------------------
# The 19 spec skills grouped into 6 functional roles (covers every skill once).
_SKILL_GROUPS = {
    'mining': ('mine_accuracy', 'mine_yield_multiplier', 'mine_cost', 'salvage_yield'),
    'offense': ('attack_power', 'attack_accuracy', 'combat_salvage_multiplier'),
    'defense': ('evade', 'shield_strength', 'shield_capacity', 'shield_efficiency'),
    'mobility': ('jump_distance', 'jump_cost', 'sensor_range'),
    'energy': ('energy_max', 'recharge_energy'),
    'negotiate': ('negotiate_skill', 'negotiate_cautious', 'negotiate_ambition'),
}

# Per-archetype tier for each group: 'favored', 'neutral' or 'reduced'. Groups
# omitted from an archetype default to 'neutral'. Every group stays reachable
# (no zero weights) so all 19 skills can still receive points and the budget is
# always fully consumed.
_SKILL_ARCHETYPES = {
    # Miner/trader: leans into mining + mobility (reach more rocks), light combat.
    'prospector': {'mining': 'favored', 'mobility': 'favored', 'offense': 'reduced'},
    # Raider: leans into offense + defense, light mining/negotiation.
    'pirate': {'offense': 'favored', 'defense': 'favored', 'mining': 'reduced', 'negotiate': 'reduced'},
    # Generalist: no lean (all groups neutral).
    'balanced': {},
}

# Tier -> relative selection weight, by bias strength. Strength is picked at
# random per allocation so loadouts range from subtle to specialist.
_SKILL_BIAS_STRENGTHS = {
    'mild': {'favored': 2.0, 'neutral': 1.0, 'reduced': 0.7},
    'moderate': {'favored': 3.0, 'neutral': 1.0, 'reduced': 0.5},
    'strong': {'favored': 5.0, 'neutral': 1.0, 'reduced': 0.3},
}

# OpponentAIType -> archetype. Types not listed fall back to 'balanced'.
_AI_TYPE_ARCHETYPE = {
    OpponentAIType.BOT_V3: 'prospector',
    OpponentAIType.BOT_V4: 'pirate',
    OpponentAIType.BOT_V7: 'prospector',
    OpponentAIType.MODEL: 'balanced',
    OpponentAIType.BOT_V2: 'balanced',
    OpponentAIType.BOT_V5: 'balanced',
}

class EnvSetupMixin:
    """Config loaders, episode generation, ship-state helpers and model loading."""

    def _load_predefined_asteroids(self) -> List[dict]:
        """
        Load predefined asteroids from configuration file.

        Config file format (JSON):
        {
            "10x10": [
                {"x": 2, "y": 3, "mass": 30, "nutrinium": 20},
                {"x": 5, "y": 7, "mass": 45, "nutrinium": 35},
                ...
            ],
            "15x15": [...],
            ...
        }

        Returns:
            List of asteroid dictionaries
        """
        import json

        # Use cached data if available
        if self._predefined_asteroids_cache is not None:
            return self._predefined_asteroids_cache

        dimension_key = f"{self.map_width}x{self.map_height}"

        try:
            if not os.path.exists(self.asteroid_config_path):
                logger.warning(f"Asteroid config file not found: {self.asteroid_config_path}")
                logger.warning(f"Falling back to random asteroid generation.")
                return None

            with open(self.asteroid_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if dimension_key not in config:
                logger.warning(f"No asteroid configuration found for dimension '{dimension_key}' in {self.asteroid_config_path}")
                logger.warning(f"Available dimensions: {list(config.keys())}")
                logger.warning(f"Falling back to random asteroid generation.")
                return None

            dimension_config = config[dimension_key]
# Support both old format (list) and new format (dict with asteroids/trading_posts)
if isinstance(dimension_config, list):
    # Old format - just a list of asteroids
    asteroids_data = dimension_config
elif isinstance(dimension_config, dict) and 'asteroids' in dimension_config:
    # New format - dict with 'asteroids' and optionally 'trading_posts'
    asteroids_data = dimension_config['asteroids']
else:
    logger.warning(f"Invalid asteroid config format for dimension '{dimension_key}'")
    logger.warning(f"  Expected list or dict with 'asteroids' key")
    logger.warning(f"  Falling back to random asteroid generation.")
    return None

# Validate asteroid data and deduplicate by (x,y)
asteroids = []
seen = set()
for i, asteroid_data in enumerate(asteroids_data):
    if not all(k in asteroid_data for k in ['x', 'y', 'mass', 'nutrinium']):
        logger.warning(f"Invalid asteroid data at index {i}: {astereoid_data}")
        continue

    # Validate coordinates
    ax = int(astereoid_data['x'])
    ay = int(astereoid_data['y'])
    if not (0 <= ax < self.map_width and 0 <= ay < self.map_height):
        logger.warning(f"Asteroid at index {i} has out-of-bounds coordinates: ({astereoid_data['x']}, {astereoid_data['y']})")
        continue

    # Ensure nutrinium doesn't exceed mass
    mass = int(astereoid_data['mass'])
    nutrinium = int(min(astereoid_data['nutrinium'], mass))

    key = (ax, ay)
    if key in seen:
        logger.warning(f"Duplicate asteroid position at index {i} ignored: ({ax},{ay})")
        continue

    seen.add(key)
    asteroids.append({
        'x': ax,
        'y': ay,
        'mass': mass,
        'nutrinium': nutrinium
    })

if not asteroids:
    logger.warning(f"No valid asteroids found in config for dimension '{dimension_key}'")
    logger.warning(f"  Falling back to random asteroid generation.")
    return None

# Cache the loaded data
self._predefined_asteroids_cache = asteroids
logger.info(f"Loaded {len(asteroids)} predefined asteroids from {self.asteroid_config_path} for dimension {dimension_key}")

return asteroids

except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in asteroid config file {self.asteroid_config_path}: {e}")
    logger.warning(f"  Falling back to random asteroid generation.")
    return None
except Exception as e:
    logger.error(f"Error loading asteroid config: {e}")
    logger.warning(f"  Falling back to random asteroid generation.")
    return None

def load_predefined_trading_posts(self) -> Optional[List[dict]]:
    """
    Load predefined trading posts from the same asteroid configuration file.

    Config file format (JSON):
    {
        "10x10": {
            "asteroids": [...],
            "trading_posts": [
                {"x": 1, "y": 1},
                {"x": 8, "y": 8},
                ...
            ]
        }
    }

    Returns:
        List of trading post dictionaries with x,y coordinates, or None if not found
    """
    import json

    # Don't cache trading posts separately - they come from same config as asteroids
    dimension_key = f"{self.map_width}x{self.map_height}"

    try:
        if not os.path.exists(self.asteroid_config_path):
            logger.info(f"No predefined trading posts (config file not found)")
            return None

        with open(self.asteroid_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if dimension_key not in config:
            logger.info(f"No predefined trading posts for dimension '{dimension_key}'")
        return None

    dimension_config = config[dimension_key]

    # Support both old format (list) and new format (dict with asteroids/trading_posts)
    if isinstance(dimension_config, list):
        # Old format - just asteroids, no trading posts defined
        logger.info(f"Config uses old format (list), no predefined trading posts")
        return None

    if 'trading_posts' not in dimension_config:
        logger.info(f"No 'trading_posts' key in config for dimension '{dimension_key}'")
        return None

    posts_data = dimension_config['trading_posts']

    # Validate trading post data
    posts = []
    seen = set()
    for i, post_data in enumerate(posts_data):
        if not all(k in post_data for k in ['x', 'y']):
            logger.warning(f"Invalid trading post data at index {i}: {post_data}")
            continue

        tx = int(post_data['x'])
        ty = int(post_data['y'])
        # Validate coordinates
        if not (0 <= tx < self.map_width and 0 <= ty < self.map_height):
            logger.warning(f"Trading post at index {i} has out-of-bounds coordinates: ({post_data['x']}, {post_data['y']})")
            continue

        key = (tx, ty)
        if key in seen:
            logger.warning(f"Duplicate trading post position at index {i} ignored: ({tx},{ty})")
            continue

        seen.add(key)
        posts.append({'x': tx, 'y': ty})

    if not posts:
        logger.info(f"No valid trading posts found in config for dimension '{dimension_key}'")
        return None

    logger.info(f"Loaded {len(posts)} predefined trading posts from {self.asteroid_config_path} for dimension {dimension_key}")
    return posts

except json.JSONDecodeError as e:
    logger.warning(f"Invalid JSON in config file (trading posts): {e}")
    return None
except Exception as e:
    logger.warning(f"Error loading predefined trading posts: {e}")
    return None


def load_predefined_start_position(self) -> Optional[dict]:
    """
    Load predefined starting position from configuration file.

    Config file format (JSON):
    {
        "10x10": {
            "player_x": 5,
            "player_y": 5
        },
        "15x15": {
            "player_x": 7,
            "player_y": 7
        },
        ...
    }

    Returns:
        Dictionary with player_x and player_y, or None if not found
    """
    import json

    # Use cached data if available
    if self._predefined_start_cache is not None:
        return self._predefined_start_cache

    dimension_key = f"{self.map_width}x{self.map_height}"

    try:
        if not os.path.exists(self.start_position_config_path):
            logger.warning(f"Start position config file not found: {self.start_position_config_path}")
            logger.warning(f"Falling back to random starting position.")
            return None

        with open(self.start_position_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if dimension_key not in config:
            logger.warning(f"No start position configuration found for dimension '{dimension_key}' in {self.start_position_config_path}")
            logger.warning(f"Available dimensions: {list(config.keys())}")
            logger.warning(f"Falling back to random starting position.")
            return None

        start_data = config[dimension_key]

        # Validate start position data
        if not all(k in start_data for k in ['player_x', 'player_y']):
logger.warning(f"Invalid start position data (missing player_x or player_y): {start_data}")
logger.warning(f"Falling back to random starting position.")
return None

# Validate coordinates
if not (0 <= start_data['player_x'] < self.map_width and 0 <= start_data['player_y'] < self.map_height):
    logger.warning(f"Start position out of bounds: ({start_data['player_x']}, {start_data['player_y']})")
    logger.warning(f"Falling back to random starting position.")
return None

start_position = {
    'player_x': int(start_data['player_x']),
    'player_y': int(start_data['player_y'])
}

# Cache the loaded data
self._predefined_start_cache = start_position
logger.info(f"Loaded predefined start position from {self.start_position_config_path} for dimension {dimension_key}: {{start_position['player_x']}, {start_position['player_y']}}")
return start_position

except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in start position config file {self.start_position_config_path}: {e}")
    logger.warning(f"Falling back to random starting position.")
return None
except Exception as e:
    logger.error(f"Error loading start position config: {e}")
    logger.warning(f"Falling back to random starting position.")
return None

def _parse_enemy_model_entry(self, raw_entry: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse enemy model config token.

    Supported formats:
    - model_path
    - model_path::SPEC_NAME
    """
    token = (raw_entry or '').strip()
    if not token:
        return None, None
    if '::' not in token:
        return token, None
    model_path, spec_name = token.split('::', 1)
    model_path = model_path.strip()
    spec_name = spec_name.strip()
    if not model_path:
        return None, None
    return model_path, (spec_name or None)

def load_enemy_model_entries(self) -> Optional[List[Dict[str, Optional[str]]]]:
    """Load parsed enemy model entries from config.

    Config file supports both formats:
    - models/v1/ppo_pnp_model_v74
    - models/v1/ppo_pnp_model_v74::DEFAULT_COMPACT_SPEC
    """
    if self.enemy_model_entries is not None:
        return self._enemy_model_entries

    try:
        if not os.path.exists(self.enemy_models_config_path):
            logger.info(f"Enemy models config file not found: {self.enemy_models_config_path}")
            logger.info("No model-based enemies will be created.")
            return None

        entries: List[Dict[str, Optional[str]]] = []
        with open(self.enemy_models_config_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                model_path, spec_name = self._parse_enemy_model_entry(line)
                if not model_path:
                    logger.warning(
                        "Ignoring invalid enemy model config entry at line %d: '%s'",
                        line_num,
                        line
                    )
                    continue

                entries.append({'path': model_path, 'spec_name': spec_name})

        if not entries:
            logger.info(f"No model paths found in {self.enemy_models_config_path}")
            logger.info("No model-based enemies will be created.")
            return None

        self._enemy_model_entries = entries
        logger.info(
            "Loaded %d enemy model entries from %s",
            len(entries),
            self.enemy_models_config_path,
        )
        return entries

    except Exception as e:
        logger.warning(f"Error loading enemy model entries: {e}")
        return None
def _load_enemy_model_paths(self) -> Optional[List[str]]:
    """
    Load enemy model paths from configuration file.

    Config file format (text file, one model path per line):
    models/ppo_v1_pnp_model
    models/ppo_v5_pnp_model
    # Lines starting with # are comments

    Returns:
        List of model paths, or None if no valid models found
    """
    # Keep backward compatibility for existing callers that only expect path strings.
    if self._enemy_model_paths is not None:
        return self._enemy_model_paths

    entries = self._load_enemy_model_entries()
    if not entries:
        return None
    self._enemy_model_paths = [entry['path'] for entry in entries if entry.get('path')]
    return self._enemy_model_paths or None


def _skill_weight_profile(self, archetype: str, strength: str) -> dict:
    """
    Build a `skill -> selection weight` map for an archetype/strength.

    Each skill inherits the weight of its group's tier ('favored' / 'neutral' / 'reduced') under the chosen ``strength``. Groups not mentioned by the archetype, and any skill not covered by a group, are treated as neutral. All weights are strictly positive so every skill remains reachable.
    """
    tier_weights = _SKILL_BIAS_STRENGTHS.get(strength, _SKILL_BIAS_STRENGTHS['moderate'])
    group_tiers = _SKILL_ARCHETYPES.get(archetype, {})

    weights = {name: tier_weights['neutral'] for name in self.config['abilities'].keys()}
    for group, skills in _SKILL_GROUPS.items():
        tier = group_tiers.get(group, 'neutral')
        weight = tier_weights[tier]
        for name in skills:
            if name in weights:
                weights[name] = weight
    return weights


def _distribute_skill_points(self, budget: int, ai_type=None) -> dict:
    """
    Distribute a skill-point budget randomly across the 19 spec skills,
    biased toward the ship's behavioural archetype.

    Every ship receives the same `budget` each episode, but the allocation is independent and random per ship. The archetype is derived from ``ai_type`` (e.g., PIRATE/BOT_V4 lean toward attack & defense, while PROSPECTOR/BOT_V3 lean toward mining & mobility); the player ship passes ``ai_type=None`` and is given a random archetype each episode so the agent trains against varied loadouts. The bias strength (mild / moderate / strong) is also chosen at random per allocation.

    Starting from all-zero, a not-yet-capped skill is drawn with probability proportional to its archetype weight and incremented by 1 until the budget is fully consumed. Each skill is capped at 10, so the effective maximum spend is `19 * 10 = 190`; budgets above that are clamped (all skills reach 10). Because every weight is positive, all 19 skills remain reachable and the returned dict's values always sum to ``min(budget, 190)``.
    """
    cap = 10
    skills = list(self.config['abilities'].keys())
    abilities = {name: 0 for name in skills}

    if ai_type is None:
        archetype = random.choice(list(_SKILL_ARCHETYPES.keys()))
    else:
        archetype = _AI_TYPE_ARCHETYPE.get(ai_type, 'balanced')
    strength = random.choice(list(_SKILL_BIAS_STRENGTHS.keys()))
    weight_profile = self._skill_weight_profile(archetype, strength)

    effective = max(0, min(int(budget), len(skills) * cap))
    available = list(skills)
    weights = [weight_profile[name] for name in available]
    for _ in range(effective):
        name = random.choices(available, weights=weights)[0]
        abilities[name] += 1
        if abilities[name] >= cap:
            idx = available.index(name)
            available.pop(idx)
            weights.pop(idx)
    return abilities


def generate_episode_modules(self) -> list:
    """
    Choose which module-locked actions are installed this episode.

    All ships share the same module set (a level playing field). Default mode 'all' installs every module so jump/repair/salvage-heavy training is unaffected; mode 'random' installs each module with 50% probability to train module-gated behaviour; 'none' installs nothing. Controlled by self.module_grant_mode.
    """
    all_modules = ['JUMP', 'REPAIR', 'SALVAGE']
    mode = getattr(self, 'module_grant_mode', 'all')
    if mode == 'random':
        return [m for m in all_modules if random.random() < 0.5]
    if mode == 'none':
def skill(self, ship: dict, name: str) -> int:
    """Return the effective level (0-10) of a skill for a ship.

    Skills live in ship['abilities']; missing skills default to 0. This is the single accessor every mechanic uses so skill effects are applied uniformly.
    """
    try:
        return int(ship.get('abilities', {}).get(name, 0) or 0)
    except (TypeError, ValueError):
        return 0

def shield_state(self, ship: dict) -> str:
    """Return the shield state string ('POWERED'|'DRAINING'|'DOWN')."""
    shield = ship.get('shield')
    if isinstance(shield, dict):
        return shield.get('state', 'DOWN')
    # Legacy fallback: shields_up flag maps to POWERED/DOWN.
    return 'POWERED' if ship.get('shields_up') else 'DOWN'

def action_allowed(self, ship: dict, action_name: str) -> bool:
    """Honor metadata.actionRestrictions for a ship's current state.

    Checks allowedWhileRecharging (when ship is recharging) and allowedWithShieldsUp (when shields are POWERED). Unknown actions default to allowed so new actions are not accidentally blocked.
    """
    restrictions = self.config.get('action_restrictions', {})
    rule = restrictions.get(action_name)
    if rule is None:
        return True
    if ship.get('recharging') and not rule.get('allowedWhileRecharging', True):
        return False
    if self._shield_state(ship) == 'POWERED' and not rule.get('allowedWithShieldsUp', True):
        return False
    return True

def has_module(self, ship: dict, module_name: str) -> bool:
    """Whether a ship has a given equipable module (e.g. JUMP, REPAIR, SALVAGE)."""
    modules = ship.get('modules')
    if not modules:
        return False
    return module_name in modules

def assign_negotiate_objectives(self) -> None:
    """Assign each ship a random negotiate objective pointing at a trading post.

    Mirrors the spec: every ship starts a round with a negotiate objective (me.objectives.negotiate -> {tradingPostName, tradingPostId}). No-op if there are no trading posts.
    """
    if not hasattr(self, 'trading_posts', None):
        return
    ships = [self.player_ship] + list(getattr(self, 'opponent_ships', []))
    for ship in ships:
        if ship is None:
            continue
        post = random.choice(self.trading_posts)
        ship.setdefault('objectives', {})[['negotiate'] = {
            'tradingPostName': post.get('name'),
            'tradingPostId': post.get('id'),
        }

def _load_enemy_model(self, model_path: str):
    """Load a trained model for enemy AI.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded compatibility-wrapped model instance, or None if loading fails
    """
    # Check cache first
    if model_path in self._enemy_models:
        return self._enemy_models[model_path]

    # Skip paths already known to be unavailable for this env instance.
    if model_path in self._enemy_model_unavailable_paths:
        return None

    # Check if stable-baselines3 is available
    if not SB3_AVAILABLE:
        self._mark_enemy_model_unavailable(model_path, "stable-baselines3 not available")
        return None

    try:
        # Try to load model without environment first to check action space
        temp_model = None

        # Try PPO first (most common)
        if 'ppo' in model_path.lower():
            try:
                temp_model = PPO.load(model_path)
            except Exception:
                pass

        # Try DQN if PPO failed
if temp_model is None and 'dqn' in model_path.lower():
    try:
        temp_model = DQN.load(model_path)
    except Exception:
        pass

# Try A2C if both failed
if temp_model is None:
    try:
        temp_model = A2C.load(model_path)
    except Exception:
        pass

# If we couldn't load the model at all, return None
if temp_model is None:
    self.mark_enemy_model_unavailable(model_path, "unable to deserialize with PPO/DQN/A2C loaders")
    return None

# Check action space compatibility. The env now uses a structured Dict action
# space (action_type + target + energy); legacy enemy models are scalar
# Discrete(14/15). Their scalar prediction is still a valid *action type* in the
# new 19-type space, so we accept them through the compat wrapper (which maps
# model action ids into the env action-type set). Structured-action models (no
# ``.n``) and oversized spaces are rejected.
model_action_space = getattr(temp_model.action_space, 'n', None)
env_action_space = self.num_action_types

if model_action_space is None:
    self.mark_enemy_model_unavailable(
        model_path, "model action space is not Discrete; structured-action models unsupported")
    return None
if model_action_space > env_action_space:
    self.mark_enemy_model_unavailable(
        model_path,
        f"incompatible action space {model_action_space} vs {env_action_space}")
    return None
if model_action_space != env_action_space:
    logger.info(
        f"Enemy model {model_path} has a smaller action space ({model_action_space} vs "
        f"{env_action_space}); using compatibility mapping")

# Reload model WITH environment binding for proper normalization and prediction.
# First check observation/action space compatibility before binding.
obs_compat = True
try:
    from gymnasium import spaces as _spaces
    # A scalar-Discrete model cannot be env-bound to the structured MultiDiscrete
    # action space, so skip binding and drive it via the compat wrapper instead.
    if not isinstance(self.action_space, _spaces.Discrete):
        obs_compat = False
        logger.info(
            "Enemy model %s is scalar-Discrete while env uses a structured action space; "
            "skipping env binding and using the compat wrapper",
            model_path,
        )
    model_obs_space = temp_model.observation_space
    if isinstance(model_obs_space, _spaces.Box) and isinstance(self.observation_space, _spaces.Dict):
        obs_compat = False
        logger.info(
            "Enemy model %s expects flat Box observation while env provides Dict; skipping env binding",
            model_path,
        )
    if (isinstance(model_obs_space, _spaces.Dict) and
            isinstance(self.observation_space, _spaces.Dict) and
            'observation' in model_obs_space.spaces and
            'observation' in self.observation_space.spaces):
        if model_obs_space['observation'].shape != self.observation_space['observation'].shape:
            obs_compat = False
            logger.info(f"Enemy model {model_path} has different obs size "
                       f"{model_obs_space['observation'].shape[0]} vs "
                       f"{self.observation_space['observation'].shape[0]}, ",
                       "skipping env binding")
except Exception:
    pass

if not obs_compat:
    wrapped_model = wrap_model_with_compat(
        temp_model,
        env_action_space_size=self.num_action_types,
        enable_action_masking=True,
    )
    self.enemy_models[model_path] = wrapped_model
    logger.info(f"Loaded enemy model without env binding: {model_path} (action space: {model_action_space})")
    return wrapped_model

try:
    # Determine which algorithm to use for reload
    final_model = None
    if 'ppo' in model_path.lower() or isinstance(temp_model, PPO):
        final_model = PPO.load(model_path, env=self)
    elif 'dqn' in model_path.lower() or isinstance(temp_model, DQN):
        final_model = DQN.load(model_path, env=self)
    else:
        final_model = A2C.load(model_path, env=self)

    wrapped_model = wrap_model_with_compat(
        final_model,
        env_action_space_size=self.num_action_types,
        enable_action_masking=True,
    )
# Cache the environment-bound wrapped model
self._enemy_models[model_path] = wrapped_model
logger.info(f"Loaded enemy model with env binding: {model_path} (action space: {model_action_space})")
return wrapped_model
except Exception as e:
    # Fallback to model without env binding if reload fails
    logger.warning(f"Failed to reload model with env binding ({e}), using original model without binding")
    wrapped_model = wrap_model_with_compat(
        temp_model,
        env_action_space_size=self.num_action_types,
        enable_action_masking=True,
    )
    self._enemy_models[model_path] = wrapped_model
    logger.info(f"Loaded enemy model without env binding: {model_path} (action space: {model_action_space})")
    return wrapped_model

except Exception as e:
    self._mark_enemy_model_unavailable(model_path, f"{e}")
return None

def _mark_enemy_model_unavailable(self, model_path: str, reason: str) -> None:
    """Cache a model path as unavailable and emit a warning only once."""
    if model_path in self._enemy_model_unavailable_paths:
        return
    self._enemy_model_unavailable_paths.add(model_path)
    logger.warning(
        "Failed to load enemy model %s: %s. Caching path as unavailable for this environment instance.",
        model_path,
        reason,
    )