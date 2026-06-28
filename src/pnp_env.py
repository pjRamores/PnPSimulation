"""
Prospectors n Pirates Game Environment
Compatible with OpenAI Gym/Gymnasium for Deep Reinforcement Learning

The environment is composed from cohesive mixin modules - setup/loaders,
action masking, observation, actions/combat, opponents/AI, geometry and
rendering. Shared imports, the module logger, the action/AI enums and the
reward classes live in ``env_common`` and are re-exported here so external
code can keep importing them from ``pnp_env``.
"""

from env_common import *
import copy
from env_setup_mixin import EnvSetupMixin
from env_masking_mixin import EnvMaskingMixin
from env_observation_mixin import EnvObservationMixin
from env_actions_mixin import EnvActionsMixin
from env_opponent_mixin import EnvOpponentMixin
from env_geometry_mixin import EnvGeometryMixin
from env_render_mixin import EnvRenderMixin
from utils import action_masker


class ProspectorsPiratesEnv(
        EnvSetupMixin,
        EnvMaskingMixin,
        EnvObservationMixin,
        EnvActionsMixin,
        EnvOpponentMixin,
        EnvGeometryMixin,
        EnvRenderMixin,
        gym.Env):
    """
    Prospectors n Pirates Game Environment

    A space mining game where ships can mine asteroids for nutrinium or attack
    other ships to steal their cargo. The goal is to maximize credits earned.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self,
                 map_width: int = 10,
                 map_height: int = 10,
                 num_opponents: int = 3,
                 max_steps: int = 1000,
                 render_mode: Optional[str] = None,
                 reward_config: Optional[RewardConfig] = None,
                 warn_on_invalid_action: bool = False,
                 use_predefined_asteroids: bool = False,
                 asteroid_config_path: str = 'asteroids.config',
                 use_predefined_start: bool = False,
                 start_position_config_path: str = 'start_positions.config',
                 enemy_models_config_path: str = 'enemy_models.config',
                 terminate_on_player_death: bool = True,
                 forced_opponent_types: Optional[list] = None,
                 map_size_range: Optional[Tuple[int, int]] = None,
                 randomize_action_restrictions: bool = False,
                 randomize_game_config: bool = False,
                 game_config_overrides: Optional[dict] = None,
                 game_config_ranges: Optional[dict] = None,
                 player_model_spec: Optional['ModelSpec'] = None,
                 partial_observability: bool = False,
                 module_grant_mode: str = 'all',
                 # Rendering options
                 cell_width: Optional[int] = None,
                 minimap_mode: bool = False,
                 minimap_radius: int = 3):
        """
        Initialize the Prospectors n Pirates environment

        Args:
            map_width: Width of the game map
            map_height: Height of the game map
            num_opponents: Number of AI opponent ships
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human' or 'rgb_array')
            reward_config: Optional custom reward configuration
            warn_on_invalid_action: If True, print warnings for invalid actions
            use_predefined_asteroids: If True, load asteroids from config file instead of random generation
            asteroid_config_path: Path to asteroid configuration file (default: 'asteroids.config')
            use_predefined_start: If True, use predefined starting positions from config file
            start_position_config_path: Path to starting position configuration file (default: 'start_positions.config')
            enemy_models_config_path: Path to enemy models configuration file (default: 'enemy_models.config')
            terminate_on_player_death: If True, episode terminates when player is destroyed (default: True for training, False for full simulation)
            map_size_range: Optional (min, max) tuple. When set, reset() samples a fresh
                SQUARE map side length (map_width == map_height) uniformly in [min, max]
                every episode, so the policy is robust to per-round map-size variation.
                Asteroid and trading-post counts scale automatically with map area, and
                the observation/action shapes are unchanged. Default None keeps the fixed
                map_width/map_height passed to the constructor.
            randomize_game_config: If True, sample the per-round-varying game-mechanic config
                (asteroid density, mass regime, payout modifier, energy/jump/shield costs,
                shield resistance, hit chance, sell price, ...) within production-derived
                ranges every reset() so the policy is robust to the real config distribution.
                Observation/action-shape-defining keys (map size, sensor_range) are never
                randomized. Default False keeps a deterministic, fixed config.
            game_config_overrides: Optional dict of dotted config keys -> values that are
                pinned every episode AFTER any randomization (e.g. to replay a real
                production metadata block during evaluation). Example:
                {'asteroid_density': 0.11, 'combat.base_shield_resistance': 0.75}.
            game_config_ranges: Optional dict merged into the default randomization ranges
                (same dotted-key format) to override/extend sampled ranges.
            player_model_spec: Optional ModelSpec selecting the PLAYER's observation
                encoding. Defaults to DEFAULT_FULL_SPEC (275-dim, sensor window 5). When
                the spec's observation_spec.sensor_range is set, the env sizes its
                observation_space from it AND (per game-mechanic coupling) overrides
                config['sensor_range'] to match (e.g. WIDE_SENSOR_SPEC -> window 10,
                595-dim, bots see range 10).
            partial_observability: When True, the PLAYER's observation and action mask
                are reconstructed from a sensor-limited ActionRequest (shared with
                BOT_V6 inference via obs_reconstruction), giving exact train/inference
                parity. Default False keeps the legacy global-visibility encoding.
            module_grant_mode: Policy for which module-locked actions (JUMP, REPAIR,
                SALVAGE) are installed each episode. 'all' (default) installs every
                module so jump/repair/salvage training is unaffected; 'random' picks the
                count uniformly from {0, 1, 2, 3} then samples that many modules (every
                count equally likely) to train module-gated behaviour;
                'none' installs nothing. All ships share the same set (level playing
                field). Does not change the observation layout.
        """
        super().__init__()

        self.map_width = map_width
        self.map_height = map_height
        # Optional per-episode SQUARE map-size randomization (width == height). When
        # set to (min, max), reset() samples a fresh side length each episode so the
        # policy is robust to the per-round map-size variation seen in real games.
        # The observation/action shapes are unchanged (only normalized obs *values*
        # depend on map size, and the spatial deltas/distances are scale-free).
        if map_size_range is not None:
            lo, hi = int(map_size_range[0]), int(map_size_range[1])
            if lo < 1 or hi < lo:
                raise ValueError(
                    f"map_size_range must be (min, max) with 1 <= min <= max, got {map_size_range}"
                )
            self.map_size_range = (lo, hi)
        else:
            self.map_size_range = None
        self.num_opponents = num_opponents
        self.forced_opponent_types = forced_opponent_types
        if forced_opponent_types:
            self.num_opponents = len(forced_opponent_types)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.terminate_on_player_death = terminate_on_player_death
        self.randomize_action_restrictions = bool(randomize_action_restrictions)
        self.randomize_game_config = bool(randomize_game_config)
        self.game_config_overrides = dict(game_config_overrides) if game_config_overrides else None
        # Rendering overrides
        self.cell_width = cell_width
        self.minimap_mode = bool(minimap_mode)
        self.minimap_radius = int(minimap_radius)

        # Asteroid configuration
        self.use_predefined_asteroids = use_predefined_asteroids
        self.asteroid_config_path = asteroid_config_path
        self._predefined_asteroids_cache = None

        # Starting position configuration
        self.use_predefined_start = use_predefined_start
        self.start_position_config_path = start_position_config_path
        self._predefined_start_cache = None

        # Enemy models configuration
        self.enemy_models_config_path = enemy_models_config_path
        self._enemy_model_entries = None  # Cache of parsed entries [{'path': str, 'spec_name': Optional[str]}]
        self._enemy_model_paths = None  # Cache for loaded model paths
        self._enemy_models = {}  # Cache for loaded model instances {path: model_instance}
        self._enemy_model_specs = {}  # Cache for model specs {path: ModelSpec}
        self._enemy_model_unavailable_paths = set()  # Paths that failed loading; skip retries

        # Per-ship model specs and observation generators
        self._ship_model_specs = {}  # Map ship id to ModelSpec (using id() as key)
        self._observation_generators = {}  # Cache of generators {spec_type: generator}

        # Module grant policy for module-locked actions ('all' | 'random' | 'none').
        if module_grant_mode not in ('all', 'random', 'none'):
            raise ValueError(
                "module_grant_mode must be one of 'all', 'random', 'none'; "
                f"got {module_grant_mode!r}"
            )
        self.module_grant_mode = module_grant_mode

        # Game configuration (spec-accurate; mirrors live metadata.* structure).
        # See docs_game/game_concepts.md and docs_game/actions.md for the source of
        # every constant below. Values match the production server defaults.
        self.config = {
            'max_energy': 100,            # metadata.shipConfig.maxEnergy
            'energy_per_recharge': 10,    # metadata.shipConfig.energyPerRecharge (+recharge_energy*2/tick)
            'max_health': 100,
            'max_jump_distance': 50,      # metadata.shipConfig.maxJumpDistance (+jump_distance*10)
            'max_nutrinium_cargo': 1000,
            'max_credits': 10000,
            # Skill-point budget allocated to EVERY ship each episode. The budget
            # is common to all ships but distributed randomly (and differently)
            # across the 19 skills per ship, fully consumed, re-rolled each reset.
            # Each skill is capped 0-10, so the effective maximum is 19*10 = 190.
            'skill_point_budget': 24,
            # Normalizes skill_points_total/spent in the observation; kept equal to
            # the budget so those (now-constant) values stay within [0, 1].
            'max_skill_points': 24,
            'energy_costs': {
                # metadata.shipConfig.energyCosts
                'mine': 10,             # reduced by mine_cost skill (min 0)
                'move': 0,              # MOVE is free (recharge while moving)
                'jump': 1,              # per unit euclidean distance
                'jump_min_cost': 75,    # jumpMinCost; reduced by jump_cost skill (-5/pt, floor 0)
                'attack': 1,            # minimum payload; ATTACK spends its payload energy
                'plunder': 5,           # PLUNDER fixed cost (even on miss)
                'negotiate': 5,         # NEGOTIATE cost
                'sell': 0,              # SELL is free
                'shield_maintenance': 1,  # per-tick cost while shields POWERED
                'shields': 1,           # legacy alias (binary-shield code path; removed in P6)
            },
            'combat': {
                'base_target_number': 0.5,    # roll-to-hit threshold (attack_accuracy -5%/pt, evade +5%/pt)
                'guaranteed_miss_chance': 0.05,
                'guaranteed_hit_chance': 0.05,
                'attack_shield_damage': 1.5,  # attackShieldDamage: shieldDmg = round(dmg*1.5)
                'base_shield_resistance': 0.75,  # combat.baseShieldResistance (prod 0.25-0.82); +shield_strength*0.05
                'shield_resistance_cap': 0.9, # ceiling for base+shield_strength resistance (skill still adds value)
                'recharge_penalty': 0.2,      # target recharging -> easier to hit
                'shield_recharge_rate': 5,    # combat.shieldRechargeRate: DRAINING decay/tick + RAISE_SHIELDS cost divisor
                'base_shield_capacity': 100,  # +shield_capacity*10
                'damage_variance': 0.5,       # +-50% damage spread (energy*0.5..1.5)
                'default_attack_payload': 20, # energy committed per ATTACK when no bin chosen (scalar action space)
            },
            'mining': {
                'payout_modifier': 0.01,      # mining.payoutModifier: max % of remaining nutrinium per mine
                'base_success_chance': 0.5,   # legacy alias (old density path; removed in P3)
                'min_payout': 1,
                'max_payout': 10,             # legacy alias
                # Beta shape for the biased-low random payout (reuses asteroid-budget skew).
                'payout_beta_alpha': 1.5,
                'payout_beta_beta': 8.0,
            },
            'market': {
                'sell_nutrinium': 98,         # metadata.market.sell.nutrinium (base; fluctuates)
                'repair': 100,                # metadata.market.buy.repair (REPAIR credit cost)
                'nutrinium_price': 98,         # legacy alias of sell_nutrinium
                'ship_cost': 100,             # legacy alias (old respawn path; removed in P8)
                # Dynamic market: price drifts toward base, dips on simultaneous sells.
                'price_recovery_rate': 0.05,  # fraction of gap to base recovered per tick
                'price_dip_per_sale': 0.02,   # fractional dip applied per sale in a tick
                'price_min_factor': 0.5,      # floor as a fraction of base price
            },
            'team_insurance': {
                # metadata.teamInsurance: respawn cost per member = base*(1+escalation*N)
                'base_cost_per_member': 5,
                'cost_escalation': 5.0,
            },
            'negotiate': {
                'base_success_chance': 0.3,   # negotiate.baseSuccessChance; negotiate_skill shifts succ up / fail down
                'base_fail_chance': 0.25,     # negotiate.baseFailChance
                'min_fail_chance': 0.02,      # negotiate.minFailChance: fail can never reach 0
                'skill_modifier_per_point': 0.02,  # negotiate.skillModifierPerPoint: succ/fail shift per negotiate_skill
                'bonus_gain': 0.05,           # team bonus increase on SUCCESS (+negotiate_ambition*10%)
                'bonus_penalty': 0.02,        # team bonus decrease on FAIL (-negotiate_caution*8%)
                'max_team_bonus': 1.0,        # diminishing-returns ceiling
            },
            'salvage': {
                'wreckage_percent': 0.5,      # portion of destroyed nutrinium left as wreckage
                'energy_cost': 3,             # metadata.salvage.energyCost
            },
            'asteroid_density': 0.11,      # production mapConfig.asteroidDensity
            # Trading posts scale with map area. ``trading_post_density`` is derived
            # from production (24 posts on a 125x125 map = 24/15625 ~= 0.0015360
            # posts/cell, ~1 per 651 cells). The per-episode target is
            #   max(trading_post_min, round(density * map_width * map_height))
            # unless ``trading_post_count`` is set to an explicit (non-None) override.
            'trading_post_count': None,         # explicit override; None -> derive from density
            'trading_post_density': 0.0015360,  # posts per cell (production: 24/15625)
            'trading_post_min': 4,              # floor for small maps
            'sensor_range': 5,              # metadata.sensors.range (prod 5-14); default obs window (WIDE_SENSOR_SPEC widens to 10)
            # Nutrinium distribution (production concentration model).
            # Each asteroid's nutrinium = round(concentration * mass), where
            #   concentration = nutrinium_min_percent
            #       + Beta(alpha, beta) * (nutrinium_max_percent - nutrinium_min_percent).
            # Total nutrinium therefore scales with the number of asteroids (and
            # hence with map area), mirroring the production mapConfig where
            # minNutriniumPercent=0.08 and maxNutriniumPercent=1.0.
            'nutrinium_min_percent': 0.08,      # production mapConfig.minNutriniumPercent
            'nutrinium_max_percent': 1.0,       # production mapConfig.maxNutriniumPercent
            # Beta distribution shape parameters for nutrinium concentration per asteroid.
            # alpha < beta -> skew toward lower concentrations (more poor asteroids).
            # With alpha=1.5, beta=8: ~30% poor, ~50% medium, ~20% rich asteroids.
            # The few rich asteroids become high-value strategic targets.
            'nutrinium_beta_alpha': 1.5,
            'nutrinium_beta_beta': 8.0,
            # Mass range for asteroids (production mapConfig.minMass/maxMass).
            'asteroid_mass_min': 50,
            'asteroid_mass_max': 500,
            # Default action restrictions (metadata.actionRestrictions). Each action maps to
            # {allowedWhileRecharging, allowedWithShieldsUp}. Used by validation/tick logic.
            # Values mirror the production game config (logs/action_request.json mapConfig).
            'action_restrictions': {
                'WAIT':          {'allowedWhileRecharging': True,  'allowedWithShieldsUp': True},
                'MINE':          {'allowedWhileRecharging': True,  'allowedWithShieldsUp': False},
                'MOVE':          {'allowedWhileRecharging': True,  'allowedWithShieldsUp': True},
                'RECHARGE':      {'allowedWhileRecharging': False, 'allowedWithShieldsUp': False},
                'RECHARGE_END':  {'allowedWhileRecharging': True,  'allowedWithShieldsUp': True},
                'ATTACK':        {'allowedWhileRecharging': False, 'allowedWithShieldsUp': True},
                'JUMP':          {'allowedWhileRecharging': False, 'allowedWithShieldsUp': True},
                'SELL':          {'allowedWhileRecharging': False, 'allowedWithShieldsUp': True},
                'RAISE_SHIELDS': {'allowedWhileRecharging': False, 'allowedWithShieldsUp': False},
                'LOWER_SHIELDS': {'allowedWhileRecharging': True,  'allowedWithShieldsUp': True},
                'PLUNDER':       {'allowedWhileRecharging': True,  'allowedWithShieldsUp': False},
                'SALVAGE':       {'allowedWhileRecharging': True,  'allowedWithShieldsUp': False},
                'REPAIR':        {'allowedWhileRecharging': False, 'allowedWithShieldsUp': True},
                'NEGOTIATE':     {'allowedWhileRecharging': False, 'allowedWithShieldsUp': True},
                'RESPAWN':       {'allowedWhileRecharging': True,  'allowedWithShieldsUp': True},
            },
            # Ship abilities max values for normalization (all 19 skills, 0-10).
            'abilities': {
                'energy_max': 10,
                'recharge_energy': 10,
                'mine_accuracy': 10,
                'mine_yield_multiplier': 10,
                'mine_cost': 10,
                'attack_power': 10,
                'attack_accuracy': 10,
                'evade': 10,
                'shield_strength': 10,
                'shield_capacity': 10,
                'shield_efficiency': 10,
                'combat_salvage_multiplier': 10,
                'jump_distance': 10,
                'jump_cost': 10,
                'sensor_range': 10,
                'salvage_yield': 10,
                'negotiate_skill': 10,
                'negotiate_caution': 10,
                'negotiate_ambition': 10,
            },
            # Observation parameters
            'top_asteroids_count': 5  # Number of top asteroids to include in observation
        }

        # Immutable baseline of the production action-restriction matrix. reset()
        # restores this each episode (or derives a randomized variant from it when
        # randomize_action_restrictions is set) so config['action_restrictions']
        # always starts from a known-good baseline.
        self._base_action_restrictions = copy.deepcopy(self.config['action_restrictions'])

        # Player model spec selects the PLAYER's observation encoding. Resolve it
        # BEFORE the base-config snapshot and the obs_size formula so that a spec
        # carrying an explicit sensor_range (e.g. WIDE_SENSOR_SPEC) drives BOTH the
        # observation_space sizing AND the game-mechanic config['sensor_range']
        # (opponent visibility + bot metadata) consistently. None -> DEFAULT_FULL_SPEC.
        self.player_model_spec = player_model_spec or DEFAULT_FULL_SPEC
        _spec_sensor_range = self.player_model_spec.observation_spec.sensor_range
        if _spec_sensor_range is not None:
            self.config['sensor_range'] = int(_spec_sensor_range)

        # Partial-observability mode: when True, the PLAYER's observation AND action
        # mask are reconstructed from a sensor-limited ActionRequest via the shared
        # obs_reconstruction module -- byte-identical to what a delegating BOT_V6 sees
        # at inference. Default False preserves the legacy global-visibility encoding
        # (backward compatible with models trained on global observations).
        self.partial_observability = bool(partial_observability)

        # Immutable baseline of the entire game config. _sample_episode_config()
        # restores from this each reset() before (optionally) randomizing the
        # per-round-varying values and/or applying explicit pins, so every episode
        # runs against one self-consistent config without drift accumulating.
        self._base_game_config = copy.deepcopy(self.config)

        # Per-episode randomization ranges, keyed by dotted config path. Values are
        # production-derived (games 37216/37217 span 6 rounds). Spec tuples:
        #   ('uniform', lo, hi)  -> random.uniform(lo, hi)
        #   ('int', lo, hi)      -> random.randint(lo, hi)
        #   ('choice', [vals])   -> random.choice(vals)
        #   ('group', [dicts])   -> pick one dict; apply all its dotted keys together
        # Observation/action-shape-defining keys (map_width/height, sensor_range,
        # top_asteroids_count, skill_point_budget) are intentionally excluded, as are
        # the values production keeps constant across rounds (jumpMinCost, plunder,
        # negotiate cost, shieldMaintenance, attackShieldDamage, rechargePenalty,
        # shieldRechargeRate, teamInsurance, salvage energyCost).
        self._game_config_ranges = {
            'asteroid_density': ('uniform', 0.02, 0.12),
            'mining.payout_modifier': ('uniform', 0.01, 0.10),
            'max_energy': ('choice', [90, 100]),
            'energy_per_recharge': ('choice', [8, 10]),
            'max_jump_distance': ('choice', [50, 60]),
            'energy_costs.mine': ('choice', [1, 10]),
            'energy_costs.jump': ('choice', [1, 2]),
            'energy_costs.shields': ('choice', [1, 2]),
            'market.sell_nutrinium': ('uniform', 97.0, 101.0),
            'combat.base_shield_resistance': ('uniform', 0.25, 0.85),
            'combat.base_target_number': ('uniform', 0.44, 0.50),
            # Correlated asteroid mass/nutrinium regime: production pairs the small-mass
            # map (1-100) with minNutriniumPercent 0.01, and the large-mass map (50-500)
            # with 0.08. Sampling them together preserves that coupling.
            '_mass_regime': ('group', [
                {'asteroid_mass_min': 1, 'asteroid_mass_max': 100, 'nutrinium_min_percent': 0.01},
                {'asteroid_mass_min': 50, 'asteroid_mass_max': 500, 'nutrinium_min_percent': 0.08},
            ]),
        }
        if game_config_ranges:
            self._game_config_ranges.update(game_config_ranges)

        # Reward shaping - configurable; allow injection of custom RewardConfig
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        # By default use the simple RewardCalculator; if user requested a composite, build it
        if getattr(self.reward_config, 'use_composite', False):
            try:
                # Import component classes locally to avoid top-level import issues
                from reward_utils import (
                    RewardCalculatorComposite,
                    DistanceToAsteroidReward,
                    SurvivalReward,
                    PenaltyNearEnemyReward,
                    SellBonusReward,
                    MiningQualityReward,
                    InappropriateActionPenalty,
                    EndOfEpisodeNutriniumReward,
                    EarlyDeathPenaltyReward,
                    CreditProgressReward,
                    IdleLoopPenalty,
                    EndOfEpisodeCreditReward,
                    PlacementReward,
                    EnergyStarvationPenalty,
                    OverriddenActionPenalty,
                )

                # Mapping of short names to constructor callables
                component_map = {
                    'DistanceToAsteroidReward': DistanceToAsteroidReward,
                    'SurvivalReward': SurvivalReward,
                    'PenaltyNearEnemyReward': PenaltyNearEnemyReward,
                    'SellBonusReward': SellBonusReward,
                    'MiningQualityReward': MiningQualityReward,
                    'InappropriateActionPenalty': InappropriateActionPenalty,
                    'EndOfEpisodeNutriniumReward': EndOfEpisodeNutriniumReward,
                    'EarlyDeathPenaltyReward': EarlyDeathPenaltyReward,
                    'CreditProgressReward': CreditProgressReward,
                    'IdleLoopPenalty': IdleLoopPenalty,
                    'EndOfEpisodeCreditReward': EndOfEpisodeCreditReward,
                    'PlacementReward': PlacementReward,
                    'EnergyStarvationPenalty': EnergyStarvationPenalty,
                    'OverriddenActionPenalty': OverriddenActionPenalty,
                }

                comps = []
                specs = self.reward_config.composite_components or []
                if specs:
                    for spec in specs:
                        try:
                            if isinstance(spec, str):
                                name = spec
                                params = {}
                            elif isinstance(spec, dict):
                                name = spec.get('name')
                                params = spec.get('params', {}) or {}
                            else:
                                continue

                            ctor = component_map.get(name)
                            if ctor is None:
                                # Unknown component name; skip
                                continue
                            # Instantiate with params if any
                            comps.append(ctor(**params) if params else ctor())
                        except Exception:
                            # Skip components that fail to construct
                            continue
                else:
                    # Default composite components - balanced to prioritize credit accumulation
                    # SurvivalReward: tiny per-step bonus for staying alive (0.001/step = 0.3 total)
                    # DistanceToAsteroidReward: gentle shaping toward mineable resources
                    # CreditProgressReward: rewards credit gains, penalizes zero-credit idling
                    # IdleLoopPenalty: penalizes getting stuck in non-productive loops
                    # EndOfEpisodeCreditReward: strong end-of-episode signal for credits (0.5 per credit)
                    # InappropriateActionPenalty: penalizes contextually bad actions (e.g. MINE with no asteroid)
                    comps = [
                        DistanceToAsteroidReward(),
                        SurvivalReward(),
                        CreditProgressReward(),
                        IdleLoopPenalty(),
                        EndOfEpisodeCreditReward(),
                        EndOfEpisodeNutriniumReward(),  # Penalizes holding unsold nutrinium at episode end
                        EarlyDeathPenaltyReward(),       # Penalizes early death proportional to remaining episode
                        InappropriateActionPenalty(),
                        # --- new components (analysis-driven fixes) ---
                        PlacementReward(),               # Strong terminal reward based on final rank vs opponents
                        SellBonusReward(),               # Reinforces the mine -> sell loop
                        EnergyStarvationPenalty(),       # Discourages low-energy drifting / chronic energy starvation
                        OverriddenActionPenalty(),       # Penalty when env had to override an infeasible chosen action
                    ]

                self.reward_calc = RewardCalculatorComposite(self.reward_config, comps)
                logger.info(f"Using RewardCalculatorComposite with components: {[c.__class__.__name__ for c in comps]}")
            except Exception as e:
                logger.warning(f"Failed to construct RewardCalculatorComposite: {e}. Falling back to simple RewardCalculator.")
                self.reward_calc = RewardCalculator(self.reward_config)
        else:
            self.reward_calc = RewardCalculator(self.reward_config)

        # Invalid action handling / diagnostics
        # If True, the env will print warnings when an invalid action is provided
        self.warn_on_invalid_action = bool(warn_on_invalid_action)
        # Counter for invalid actions encountered (normalization failures or OOB)
        self.invalid_action_count = 0
        # Counter for state-invalid actions (valid action code but invalid given current state)
        self.state_invalid_action_count = 0

        # Action space: structured MultiDiscrete faithful to the game server, and
        # trainable by stable-baselines3 (which supports MultiDiscrete but not Dict action
        # spaces). The policy emits a length-3 vector:
        #   [action_type, target_slot, energy_bin]
        # - action_type: 0..18 (see ActionType)
        # - target_slot: 0..top_asteroids_count-1, an INDEX into the top-N richest
        #                asteroids the observation already exposes (entity-slot /
        #                pointer action space). JUMP_TO_ASTEROID jumps to the asteroid
        #                in the chosen slot; trading-post jumps use the dedicated
        #                JUMP_TO_TRADING_POST action (auto-targets the nearest post), so
        #                no post slot is needed. Map-size invariant: the slot index does
        #                not depend on map_width/map_height.
        # - energy_bin:  0 = "use action default", 1..N = 10%..100% of max energy (ATTACK payload)
        # A bare scalar int action is still accepted by `_normalize_action` (auto-target,
        # default energy), so the heuristic AIs and scalar-Discrete callers work unchanged.
        self.num_action_types = len(ActionType)   # 19 discrete action types
        self.num_target_slots = self.config['top_asteroids_count']  # entity slots = top-N asteroids
        self.energy_bins = 11                      # 0 = default payload, 1..10 = 10%..100%
        self.action_space = spaces.MultiDiscrete(
            [self.num_action_types, self.num_target_slots, self.energy_bins]
        )

        # Observation space: flattened representation of game state
        # Enhanced ship state (based on full metadata):
        #   - Basic: x, y, energy, health, nutrinium, credits (6)
        #   - State flags: recharging, shields_up, state_ready (3)
        #   - Skill points: total, spent (2)
        #   - Abilities: 12 ability values
        #   - Action counter: actions taken this episode (1)
        # Total ship state: 24 values
        #
        # + Strategic context (high-signal features): 8 values
        #   - at_asteroid: 1 if on asteroid with nutrinium, 0 otherwise
        #   - at_trading_post: 1 if on trading post, 0 otherwise
        #   - has_nutrinium: nutrinium / cargo_cap (how full is cargo)
        #   - enemy_in_zone: 1 if enemy at same location, 0 otherwise
        #   - nearest_asteroid_dist: distance / map_diag (normalized)
        #   - nearest_trading_post_dist: distance / map_diag (normalized)
        #   - energy_ratio: energy / max_energy (redundant but grouped with context)
        #   - episode_progress: current_step / max_steps
        #
        # + Local sensor data (grid around ship)
        # + Top 5 asteroids (x, y, mass, nutrinium, distance, score) = 6 values each = 30 total
        # + Nearest trading post (x, y, distance) = 3 values
        # + Two enemy types at player location:
        #   - Strongest enemy (x, y, energy, health, nutrinium, credits, combat_score, same_team) = 8 values
        #   - Weakest enemy (x, y, energy, health, nutrinium, credits, combat_score, same_team) = 8 values
        # Total enemy info: 16 values

        ship_state_size = 24  # Complete ship state (including action counter)
        strategic_context_size = 8  # High-signal strategic features
        sensor_grid_size = (2 * self.config['sensor_range'] + 1) ** 2
        # FULL_NO_GRID layout: identical to FULL but the local sensor-grid block is
        # dropped entirely, so the observation_space Box must exclude it. Detected via
        # the player spec's observation_type (the generator omits the grid to match).
        if self.player_model_spec.observation_spec.observation_type == 'full_no_grid':
            sensor_grid_size = 0
        top_asteroids_size = self.config['top_asteroids_count'] * 6  # 5 asteroids * 6 features each
        # Spec-fidelity features (appended so legacy offsets stay stable):
        #   7 new skills + 5 shield-state + 3 modules + 3 team/economy
        #   + 3 negotiate-objective + 3 nearest-wreckage
        spec_fidelity_size = 24

        trading_post_size = 3   # Nearest trading post (x, y, distance)
        enemy_info_size = 16    # Strongest + weakest enemy at player location (8 each, incl. same_team)

        # Action-restriction matrix (appended last): 2 flags per action id
        # (allowedWhileRecharging, allowedWithShieldsUp) so the policy can adapt to
        # dynamic/randomized restrictions instead of treating them as constant.
        action_restriction_size = 2 * self.num_action_types

        # Temporal/spatial features (appended at the very end so legacy offsets stay
        # stable): remaining_time_fraction (game time left, 1.0 -> 0.0) and
        # quadrant_norm (player's cell in a 3x3 map grid as a single normalized index).
        temporal_spatial_size = 2

        # Prey enemies (appended after temporal/spatial so legacy offsets stay stable):
        # the top 3 weakest huntable enemies -- non-teammate, weaker attack AND defense
        # than the player, sensor-visible, and holding nutrinium -- each as
        # (x, y, nutrinium) = 3 values. Lets the policy chase prey when no mineable
        # nutrinium is available. 3 prey * 3 = 9 values (zero-padded if fewer).
        prey_info_size = 9

        obs_size = (
            ship_state_size +
            strategic_context_size +
            sensor_grid_size +
            top_asteroids_size +
            trading_post_size +
            enemy_info_size +
            spec_fidelity_size +
            action_restriction_size +
            temporal_spatial_size +
            prey_info_size
        )

        # The block sum above describes the FULL observation family (FULL,
        # WIDE_SENSOR via a wider sensor_grid_size, and FULL_NO_GRID via
        # sensor_grid_size=0). The COMPACT and SENSOR_ONLY generators emit a
        # different, smaller layout, so size the observation_space Box from THEIR
        # layout instead -- otherwise reset() would return an observation whose
        # shape disagrees with observation_space. Both reuse the same config-driven
        # block sizes already computed above (top_asteroids_count, sensor_range), so
        # there are no magic numbers and the sensor-window coupling is preserved.
        _player_obs_type = self.player_model_spec.observation_spec.observation_type
        if _player_obs_type == 'compact':
            # CompactObservationGenerator: 8 ship essentials + top-asteroids block
            # + nearest trading post (3) + strongest/weakest enemy block.
            obs_size = 8 + top_asteroids_size + trading_post_size + enemy_info_size
        elif _player_obs_type == 'sensor_only':
            # SensorOnlyObservationGenerator: 6 ship essentials + local sensor grid.
            obs_size = 6 + (2 * self.config['sensor_range'] + 1) ** 2

        # Use Dict observation space to support action masking
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-1.0,  # Allow -1.0 for out-of-bounds indicators in sensor grid
                high=1.0,
                shape=(obs_size,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0,
                high=1,
                shape=(self.num_action_types,),  # One mask value per action type
                dtype=np.int8
            )
        })

        # Initialize state variables
        self.current_step = 0
        self.action_counter = 0  # Track actions taken in current episode (max ~300 for 5 min @ 1 action/sec)
        self.player_ship = None
        self.opponent_ships = []
        self.asteroids = []
        self.trading_posts = []

        # Track last actions for display
        self.last_player_action = None
        self.last_opponent_actions = {}
        # Track last action results for display (dicts)
        self.last_player_action_result = None
        self.last_opponent_action_results = {}

    def _compute_trading_post_target(self) -> int:
        """Number of trading posts to place this episode, scaled by map area.

        Uses ``trading_post_count`` as an explicit override when set (non-None);
        otherwise derives the count from ``trading_post_density`` (posts per cell)
        and clamps to at least ``trading_post_min``. Production reference: 24 posts
        on a 125x125 map -> density ~= 0.0015360 (1 per 651 cells).
        """
        explicit = self.config.get('trading_post_count')
        if explicit is not None:
            return max(0, int(explicit))
        density = self.config.get('trading_post_density', 0.0015360)
        floor = self.config.get('trading_post_min', 4)
        derived = round(self.map_width * self.map_height * density)
        return max(floor, int(derived))

    def _generate_episode_action_restrictions(self) -> dict:
        """Derive a randomized action-restriction matrix from the baseline.

        Randomly flips the ``allowedWhileRecharging`` / ``allowedWithShieldsUp``
        flags of most actions so the policy must read the encoded restriction
        features to adapt. A small set of invariants is preserved to guarantee the
        episode stays solvable (no deadlocks):

        * WAIT is always allowed (idle fallback).
        * RECHARGE_END.allowedWhileRecharging stays True (must be able to stop
          recharging) and RECHARGE.allowedWhileRecharging stays False.
        * MOVE.allowedWhileRecharging stays True (recharge-while-moving mobility).
        * RESPAWN stays fully unrestricted (destroyed-ship recovery).
        """
        restrictions = copy.deepcopy(self._base_action_restrictions)
        # Per-flag invariants that must not be randomized: {key: {flag: value}}.
        locked = {
            'WAIT': {'allowedWhileRecharging': True, 'allowedWithShieldsUp': True},
            'RECHARGE': {'allowedWhileRecharging': False},
            'RECHARGE_END': {'allowedWhileRecharging': True},
            'MOVE': {'allowedWhileRecharging': True},
            'RESPAWN': {'allowedWhileRecharging': True, 'allowedWithShieldsUp': True},
        }
        for key, rule in restrictions.items():
            for flag in ('allowedWhileRecharging', 'allowedWithShieldsUp'):
                if flag in locked.get(key, {}):
                    rule[flag] = locked[key][flag]
                else:
                    rule[flag] = bool(random.getrandbits(1))
        return restrictions

    def _set_config_value(self, dotted_key: str, value) -> None:
        """Set a (possible nested) config value addressed by a dotted key.

        e.t. ``_set_config_value('combat.base_shield_resistance', 0.8)`` writes
        ``self.cofig['combat']['base_shield_resistance'] = 0.8``. Missing
        intermediate keys are skipped silently so unknown overrides are no-ops.
        """
        parts = dotted_key.split('.')
        node = self.config
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                return
            node = node[part]
        if isinstance(node, dict):
            node[parts[-1]] = value

    def _sample_episode_config(self) -> None:
        """Restore the baseline game config, then (optionally) randomize and/or pin it.

        Called at the top of reset() before asteroids/posts/ships are generated so the
        whole episode runs against one self-consistent config. Order of precedence:

        1. Restore from the immutable production baseline (``_base_game_config``).
        2. If ``randomize_game_config``: sample each ranged key via the env's seeded
           ``random`` module (deterministic given the reset seed).
        3. If ``gamme_config_overrides``: apply explicit pins last sothey always win
           (used to replay a real production metadata block during evaluation).

        Observation/action-shape-defining keys (map size, sensor_range,
        top_asteroids_count) are never sampled, so the spaces stay fixed.
        """
        # Always start each episode from the immutable production baseline. Reassigning
        # self.config is safe because every consumer reads it live (no cached copies).
        self.config = copy.deepcopy(self._base_game_config)

        if self.randomize_game_config:
            for key, spec in self._game_config_ranges.items():
                kind = spec[0]
                if kind == 'uniform':
                    self._set_config_value(key, random.uniform(spec[1], spec[2]))
                elif kind == 'int':
                    self._set_config_value(key, random.randint(spec[1], spec[2]))
                elif kind == 'group':
                    chosen = random.choice(spec[1])
                    for group_key, group_val in chosen.items():
                        self._set_config_value(group_key, group_val)

        if self.game_config_overrides:
            for key, value in self.game_config_overrides.items():
                self._set_config_value(key, value)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0

        # Sample this episode's SQUAre map size (width == height) when randomization is
        # enabled. Must run before _sample_episode_config() and any asteroid/trading-
        # post/ship generation so the whole episode is self-consistent; asteroid and
        # trading-post counts scale automatically with map area. Uses the env-seeded
        # random module so the size is deterministic given the reset seed.
        if self.map_size_range is not None:
            side = random.randint(self.map_size_range[0], self.map_size_range[1])
            self.map_width = side
            self.map_height = side

        # Sample this episode's game config (restore baseline + optional randomization
        # and pins). Must run before any asteroid/trading-post/ship generation so the
        # whole episode is self-consistent. action_restrictions are (re)set just below.
        self._sample_episode_config()

        # Reset action tracking
        self.action_counter = 0  # Reset action counter for new episode
        self.last_player_action = None
        self.last_opponent_actions = {}
        self.last_player_action_result = None
        self.last_opponent_action_results = {}

        # Set this episode's action-restriction matrix. Restore the production
        # baseline by default, or derive a randomized variant when enabled so the
        # policy must read the restriction features to adapt. Updating config here
        # keeps the masker, the observation encoding, and the metadata sent to bots
        # all consistent for the whole episode.
        if self.randomize_action_restrictions:
            self.config['action_restrictions'] = self._generate_episode_action_restrictions()
        else:
            self.config['action_restrictions'] = copy.deepcopy(self._base_action_restrictions)

        # Initialize player ship position
        if self.use_predefined_start:
            start_pos = self._load_predefined_start_position()
            if start_pos is not None:
                player_x, player_y = start_pos['player_x'], start_pos['player_y']
            else:
                # Fallback to random if loading failed
                player_x = random.randint(0, self.map_width - 1)
                player_y = random.randint(0, self.map_height - 1)
        else:
            player_x = random.randint(0, self.map_width - 1)
            player_y = random.randint(0, self.map_height - 1)

        # Skill-point budget for this episode. Every ship gets the same budget but
        # distributes it randomly (and differently) across the 19 skills, fully
        # consumed and re-rolled each reset. Player and each opponent get their own
        # independent allocation below.
        skill_point_budget = int(self.config.get('skill_point_budget', 0))
        # Modules installed this episode (shared by all ships). Default: all module-locked
        # actions available, so jump/repair/salvage-heavy training is unaffected. Set
        # self.module_grant_mode='random' to train module-gated behaviour.
        self._episode_modules = self._generate_episode_modules()
        # Team / market / wreckage bookkeeping (spec: team-shared respawn + negotiate bonus).
        self.team_respawn_counts = {tid: 0 for tid in range(4)}
        self.team_bonuses = {tid: 0.0 for tid in range(4)}
        self.market_price = float(self.config['market']['sell_nutrinium'])
        self.wreckage = []  # list of {name, x, y, nutrinium}

        # Player's own randomized skill allocation + derived per-ship shield capacity.
        player_abilities = self._distribute_skill_points(skill_point_budget)
        player_shield_capacity = (
            self.config['combat']['base_shield_capacity']
            + player_abilities.get('shield_capacity', 0) * 10
        )

        self.player_ship = {
            'name': 'P',  # Player ship name
            'x': player_x,
            'y': player_y,
            'energy': self.config['max_energy'],
            'health': self.config['max_health'],
            'nutrinium': 0,
            'credits': 0,
            'recharging': False,
            'just_recharged': False,
            'shields_up': False,  # legacy alias of (shield['value'] > 0)
            'destroyed': False,
            'state': 'READY',  # READY, RECHARGING, DESTROYED
            'respawn_count': 0,  # Track number of times respawned this episode
            'skill_points_total': skill_point_budget,
            'skill_points_spent': sum(player_abilities.values()),
            # Ship abilities - this ship's own randomized allocation of the budget
            'abilities': player_abilities,
            # Spec-accurate fields (P0)
            'team_id': 0,
            'modules': list(self._episode_modules),
            'shield': {'state': 'DOWN', 'capacity': player_shield_capacity, 'value': 0},
            'objectives': {'negotiate': None},  # assigned after trading posts are built
        }

        # Assign player model spec
        self._set_ship_model_spec(self.player_ship, self.player_model_spec)

        # Initialize opponent ships
        self.opponent_ships = []

        # Load available enemy models
        enemy_model_entries = self._load_enemy_model_entries()

        for i in range(self.num_opponents):
            # Decide AI type: use forced list if provided, otherwise random
            use_model = False
            model_path = None
            model_spec = DEFAULT_FULL_SPEC
            ai_type = None

            if self.forced_opponent_types and i < len(self.forced_opponent_types):
                # Use forced opponent type from the list
                forced_type = self.forced_opponent_types[i]
                # forced_type is a string: BOT_V2..BOT_V6, or a model path
                forced_upper = forced_type.upper()
                if forced_upper == 'BOT_V2':
                    ai_type = OpponentAIType.BOT_V2
                elif forced_upper == 'BOT_V3':
                    ai_type = OpponentAIType.BOT_V3
                elif forced_upper == 'BOT_V4':
                    ai_type = OpponentAIType.BOT_V4
                elif forced_upper == 'BOT_V5':
                    ai_type = OpponentAIType.BOT_V5
                elif forced_upper == 'BOT_V6':
                    ai_type = OpponentAIType.BOT_V6
                elif forced_upper == 'BOT_V7':
                    ai_type = OpponentAIType.BOT_V7
                elif forced_upper == 'BOT_V8':
                    ai_type = OpponentAIType.BOT_V8
                else:
                    # Treat as model path or "model_path::SPEC_NAME"
                    ai_type = OpponentAIType.MODEL
                    use_model = True
                    parsed_path, parsed_spec_name = self._parse_enemy_model_entry(forced_type)
                    if parsed_path is None:
                        parsed_path = forced_type
                    model_path = parsed_path
                    resolved_spec = get_named_model_spec(parsed_spec_name) if parsed_spec_name else None
                    if parsed_spec_name and resolved_spec is None:
                        logger.warning(
                            "Unknown model spec '%s' for forced opponent entry '%s'. Falling back to DEFAULT_FULL_SPEC.",
                            parsed_spec_name,
                            forced_type,
                        )
                    if resolved_spec is not None:
                        model_spec = resolved_spec
            else:
                # No forced type - use random selection
                if enemy_model_entries and len(enemy_model_entries) > 0:
                    if random.random() < 0.15:  # 15% chance to use model (reduced from 50% for training performance)
                        use_model = True
                        selected = random.choice(enemy_model_entries)
                        model_path = selected['path']
                        spec_name = selected.get('spec_name')
                        resolved_spec = get_named_model_spec(spec_name) if spec_name else None
                        if spec_name and resolved_spec is None:
                            logger.warning(
                                "Unknown model spec '%s' for enemy model '%s'. Falling back to DEFAULT_FULL_SPEC.",
                                spec_name,
                                model_path,
                            )
                        if resolved_spec is not None:
                            model_spec = resolved_spec
                        ai_type = OpponentAIType.MODEL

                if ai_type is None:
                    # Randomly assign an algorithm-based AI type, chosen uniformly
                    # from every OpponentAIType except MODEL (needs a model path,
                    # handled in the enemy-model branch above) and the model-backed
                    # bots BOT_V6 / BOT_V8 (per-tick model inference and the
                    # one-time model load materially slow training). Driving this
                    # off the enum means any new non-model opponent type added in
                    # the future is automatically included in training.
                    _excluded = (
                        OpponentAIType.MODEL,
                        OpponentAIType.BOT_V6,
                        OpponentAIType.BOT_V8,
                    )
                    algorithmic_ai_types = [t for t in OpponentAIType if t not in _excluded]
                    ai_type = random.choice(algorithmic_ai_types)

            # This opponent's own randomized skill allocation + derived shield capacity.
            opp_abilities = self._distribute_skill_points(skill_point_budget, ai_type)
            opp_shield_capacity = (
                self.config['combat']['base_shield_capacity']
                + opp_abilities.get('shield_capacity', 0) * 10
            )

            ship = {
                'name': f'E{i+1}',  # Enemy ship name (E1, E2, E3, etc.)
                'id': f'opponent_{i}',
                'ai_type': ai_type,  # Store AI behavior type
                'model_path': model_path if use_model else None,  # Model path for MODEL type enemies
                'x': random.randint(0, self.map_width - 1),
                'y': random.randint(0, self.map_height - 1),
                'energy': self.config['max_energy'],
                'health': self.config['max_health'],
                'nutrinium': 0,  # Start with 0 nutrinium (must mine to collect)
                'credits': 0,    # Start with 0 credits (must sell to earn)
                'recharging': False,
                'just_recharged': False,
                'shields_up': False,  # legacy alias of (shield['value'] > 0)
                'destroyed': False,
                'state': 'READY',
                'respawn_count': 0,
                'skill_points_total': skill_point_budget,
                'skill_points_spent': sum(opp_abilities.values()),
                # Ship abilities - this ship's own randomized allocation of the budget
                'abilities': opp_abilities,
                # Spec-accurate fields (P0)
                'team_id': (i + 1) % 4,
                'modules': list(self._episode_modules),
                'shield': {'state': 'DOWN', 'capacity': opp_shield_capacity, 'value': 0},
                'objectives': {'negotiate': None},  # assigned after trading posts are built
            }
            self.opponent_ships.append(ship)

            # Assign enemy model spec from config (or default to full observation)
            self._set_ship_model_spec(ship, model_spec)

        # Generate asteroids
        self.asteroids = []

        if self.use_predefined_asteroids:
            # Load predefined asteroids from config file
            predefined = self._load_predefined_asteroids()
            if predefined is not None:
                # Deep copy the predefined asteroids to avoid modifying cached data
                self.asteroids = copy.deepcopy(predefined)
            else:
                # Fallback to random generation if loading failed
                self.use_predefined_asteroids = False  # Disable for this episode

        if not self.use_predefined_asteroids or not self.asteroids:
            # Random asteroid generation using jittered-grid (stratified sampling)
            # This reduces clustering by partitioning the map into k x k buckets
            # and placing at most one asteroid per selected bucket at a random
            # location inside that bucket.
            num_cells = max(1, self.map_width * self.map_height)
            desired = int(self.map_width * self.map_height * self.config['asteroid_density'])
            num_asteroids = max(0, min(desired, num_cells))

            if num_asteroids == 0:
                self.asteroids = []
            else:
                # Determine grid dimension k so that k*k >= num_asteroids
                k = int(math.ceil(math.sqrt(num_asteroids)))

                # Compute cell width/height (may be fractional)
                cell_w = float(self.map_width) / k
                cell_h = float(self.map_height) / k

                # Build list of bucket indices (row, col) then randomly choose num_asteroids buckets
                buckets = [(r, c) for r in range(k) for c in range(k)]
                # If more buckets than needed, sample without replacement to choose buckets
                chosen_buckets = random.sample(buckets, k=num_asteroids) if num_asteroids < len(buckets) else buckets

                placed = set()
                for (r, c) in chosen_buckets:
                    # Determine integer ranges for this bucket
                    x_min = int(math.floor(c * cell_w))
                    x_max = int(math.floor((c + 1) * cell_w)) - 1
                    y_min = int(math.floor(r * cell_h))
                    y_max = int(math.floor((r + 1) * cell_h)) - 1

                    # Clamp ranges to map bounds
                    x_min = max(0, min(self.map_width - 1, x_min))
                    x_max = max(0, min(self.map_width - 1, max(x_min, x_max)))
                    y_min = max(0, min(self.map_height - 1, y_min))
                    y_max = max(0, min(self.map_height - 1, max(y_min, y_max)))

                    # If the bucket ended up empty (very small map), fall back to global random pos
                    if x_min > x_max or y_min > y_max:
                        ax = random.randint(0, self.map_width - 1)
                        ay = random.randint(0, self.map_height - 1)
                    else:
                        ax = random.randint(x_min, x_max)
                        ay = random.randint(y_min, y_max)

                    # Ensure uniqueness; if occupied, try a few nearby cells, otherwise skip
                    attempts = 0
                    while (ax, ay) in placed and attempts < 8:
                        ax = min(self.map_width - 1, max(0, ax + random.randint(-1, 1)))
                        ay = min(self.map_height - 1, max(0, ay + random.randint(-1, 1)))
                        attempts += 1
                    if (ax, ay) in placed:
                        # As a last resort, find any free cell. Rejection-sample first
                        # (O(1) expected on sparse maps), then fall back to a single
                        # scan only if sampling keeps colliding (near-full map).
                        found_free = False
                        for _ in range(64):
                            rx = random.randint(0, self.map_width - 1)
                            ry = random.randint(0, self.map_height - 1)
                            if (rx, ry) not in placed:
                                ax, ay = rx, ry
                                found_free = True
                                break
                        if not found_free:
                            for xx in range(self.map_width):
                                found = False
                                for yy in range(self.map_height):
                                    if (xx, yy) not in placed:
                                        ax, ay = xx, yy
                                        found = True
                                        break
                                if found:
                                    break

                    placed.add((ax, ay))

                    # Mass is assigned independently; nutrinium will be set in the
                    # budget-distribution pass below.
                    mass_min = self.config.get('asteroid_mass_min', 10)
                    mass_max = self.config.get('asteroid_mass_max', 50)
                    mass = random.randint(mass_min, mass_max)
                    self.asteroids.append({'x': ax, 'y': ay, 'mass': mass, 'nutrinium': 0})

                # --- Nutrinium concentration distribution across asteroids ---
                # Production concentration model: each asteroid receives
                #   nutrinium = round(concentration * mass)
                # where concentration is drawn per asteroid as
                #   min_pct + Beta(alpha, beta) * (max_pct - min_pct).
                # Beta(1.5, 8) is right-skewed (many poor asteroids, few rich),
                # and the [min_pct, max_pct] band matches the production mapConfig
                # (minNutriniumPercent / maxNutriniumPercent). Because the value is
                # tied to each asteroid's mass, the total nutrinium scales with the
                # number of asteroids -- and therefore with map area.
                if self.asteroids:
                    alpha = self.config.get('nutrinium_beta_alpha', 1.5)
                    beta_param = self.config.get('nutrinium_beta_beta', 8.0)
                    min_pct = self.config.get('nutrinium_min_percent', 0.08)
                    max_pct = self.config.get('nutrinium_max_percent', 1.0)
                    pct_span = max(0.0, max_pct - min_pct)

                    for asteroid in self.asteroids:
                        concentration = min_pct + random.betavariate(alpha, beta_param) * pct_span
                        concentration = min(max_pct, max(0.0, concentration))
                        nutr = int(round(concentration * asteroid['mass']))
                        asteroid['nutrinium'] = min(asteroid['mass'], max(0, nutr))

        # Generate trading posts
        self.trading_posts = []

        if self.use_predefined_asteroids:
            # Load predefined trading posts from the same config file
            predefined_posts = self._load_predefined_trading_posts()
            if predefined_posts is not None:
                # Deep copy the predefined trading posts
                self.trading_posts = copy.deepcopy(predefined_posts)

        # Ensure asteroids and trading posts do not overlap and that trading posts are unique
        asteroid_positions = {(a['x'], a['y']) for a in self.asteroids}

        # If we have predefined trading posts loaded, remove any that overlap asteroids
        if self.trading_posts:
            filtered_posts = []
            seen_posts = set()
            for post in self.trading_posts:
                key = (post['x'], post['y'])
                if key in asteroid_positions:
                    logger.warning(f"Predefined trading post at {key} overlaps an asteroid and will be ignored.")
                    continue
                if key in seen_posts:
                    logger.warning(f"Duplicate predefined trading post at {key} ignored.")
                    continue
                seen_posts.add(key)
                filtered_posts.append(post)
            self.trading_posts = filtered_posts

        # If not enough trading posts (or none), generate remaining using jittered-grid
        # to ensure good spatial coverage and avoid clustering. Trading posts will
        # never overlap asteroids or each other.
        self.trading_post_target = self._compute_trading_post_target()
        needed = self.trading_post_target - len(self.trading_posts)
        if needed > 0:
            # Build set of occupied positions (asteroids + already placed posts)
            occupied = set(asteroid_positions) | {(p['x'], p['y']) for p in self.trading_posts}

            # If the map is small or needed is large, fall back to random sampling
            num_cells = self.map_width * self.map_height
            if needed >= num_cells - len(occupied):
                # Not enough free cells or heavy filling; sample from available
                available = [(x, y) for x in range(self.map_width) for y in range(self.map_height) if (x, y) not in occupied]
                if needed > len(available):
                    logger.warning(f"Not enough free cells to place {needed} additional trading posts; only {len(available)} available. Placing as many as possible.")
                    needed = len(available)
                chosen_posts = random.sample(available, k=needed) if needed > 0 else []
                for (tx, ty) in chosen_posts:
                    self.trading_posts.append({'x': tx, 'y': ty})
            else:
                # Use stratified placement: choose `needed` buckets across a k x k grid
                # where k*k >= needed, then pick a random location inside each bucket
                k = int(math.ceil(math.sqrt(needed)))
                cell_w = float(self.map_width) / k
                cell_h = float(self.map_height) / k

                buckets = [(r, c) for r in range(k) for c in range(k)]
                chosen_buckets = random.sample(buckets, k=needed) if needed < len(buckets) else buckets

                placed_posts = set((p['x'], p['y']) for p in self.trading_posts)

                for (r, c) in chosen_buckets:
                    # Determine integer ranges for this bucket
                    x_min = int(math.floor(c * cell_w))
                    x_max = int(math.floor((c + 1) * cell_w)) - 1
                    y_min = int(math.floor(r * cell_h))
                    y_max = int(math.floor((r + 1) * cell_h)) - 1

                    # Clamp ranges to map bounds
                    x_min = max(0, min(self.map_width - 1, x_min))
                    x_max = max(0, min(self.map_width - 1, max(x_min, x_max)))
                    y_min = max(0, min(self.map_height - 1, y_min))
                    y_max = max(0, min(self.map_height - 1, max(y_min, y_max)))

                    # Choose a random candidate inside bucket avoiding occupied cells
                    found = False
                    tries = 0
                    while not found and tries < 12:
                        if x_min > x_max or y_min > y_max:
                            tx = random.randint(0, self.map_width - 1)
                            ty = random.randint(0, self.map_height - 1)
                        else:
                            tx = random.randint(x_min, x_max)
                            ty = random.randint(y_min, y_max)

                        if (tx, ty) in occupied or (tx, ty) in placed_posts:
                            # try nearby
                            tx = min(self.map_width - 1, max(0, tx + random.randint(-1, 1)))
                            ty = min(self.map_height - 1, max(0, ty + random.randint(-1, 1)))
                            tries += 1
                            continue

                        # Accept this post
                        self.trading_posts.append({'x': tx, 'y': ty})
                        placed_posts.add((tx, ty))
                        found = True

                    if not found:
                        # Fallback: find any available cell. Rejection-sample first
                        # (cheap on sparse maps), then scan once only if still colliding.
                        for _ in range(64):
                            rx = random.randint(0, self.map_width - 1)
                            ry = random.randint(0, self.map_height - 1)
                            if (rx, ry) not in occupied and (rx, ry) not in placed_posts:
                                self.trading_posts.append({'x': rx, 'y': ry})
                                placed_posts.add((rx, ry))
                                found = True
                                break
                        if not found:
                            for xx in range(self.map_width):
                                for yy in range(self.map_height):
                                    if (xx, yy) not in occupied and (xx, yy) not in placed_posts:
                                        self.trading_posts.append({'x': xx, 'y': yy})
                                        placed_posts.add((xx, yy))
                                        found = True
                                        break
                                if found:
                                    break
                        if not found:
                            logger.warning('Unable to place an expected trading post due to lack of free cells.')

        # Give every trading post a stable name/id (used by NEGOTIATE objectives) and
        # assign each ship an initial negotiate objective pointing at one of them.
        for idx, post in enumerate(self.trading_posts):
            post.setdefault('id', f'tp_{idx}')
            post.setdefault('name', f'TP-{idx}')
        self._assign_negotiate_objectives()

        # Build spatial lookup cache for fast entity-at-location queries.
        # Asteroid/trading-post positions are static for the whole episode, so
        # invalidate and rebuild their index now; per-step refreshes only opponents.
        self._static_cache_ready = False
        self._rebuild_location_cache()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, dict]:
        """Execute one step in the environment"""
        # Rebuild spatial lookup cache for fast entity-at-location queries
        self._rebuild_location_cache()
        # Market price drifts back toward its base each tick (sales dip it).
        self._update_market()
        # Per-ship tick order (steps 1-2): apply shield maintenance + recharge gain
        # to the player BEFORE its action so recharge energy is spendable this tick
        # and validity/mask checks below see the post-recharge energy.
        self._pre_action_tick(self.player_ship)
        # Preserve the raw requested action for debugging
        requested_action_raw = action

        # Normalize action -> (action_type, target, energy). Scalar/legacy inputs yield
        # (atype, None, None) so existing callers and AIs keep their auto-target behaviour.
        action_target = None
        action_energy = None
        try:
            action, action_target, action_energy = self._normalize_action(action)
            action_valid = 0 <= action < self.num_action_types
        except Exception as e:
            # Normalization failed (e.g., non-int-like input). Default to WAIT
            self.invalid_action_count += 1
            if self.warn_on_invalid_action:
                logger.warning(f"Unable to normalize action {requested_action_raw!r}: {e}. Defaulting to WAIT.")
            action = int(ActionType.WAIT)
            action_valid = False

        # Apply action mapping from player's model spec
        # This allows models with different action space sizes to compete
        player_spec = self._get_ship_model_spec(self.player_ship)
        mapped_action = player_spec.action_spec.map_action(action, env_action_space_size=self.num_action_types)
        if action != mapped_action:
            action = mapped_action

        # Validate bounds; if out of range, default to WAIT
        if not (0 <= action < self.num_action_types):
            self.invalid_action_count += 1
            if self.warn_on_invalid_action:
                logger.warning(f"Action {action} out of bounds [0, {self.num_action_types - 1}]. Defaulting to WAIT.")
            action = int(ActionType.WAIT)
            action_valid = False

        # If player is destroyed, force RESPAWN action
        if self.player_ship['destroyed']:
            if action != ActionType.RESPAWN:
                # Override any other action to RESPAWN
                if self.warn_on_invalid_action:
                    logger.warning(f"Player is destroyed. Only RESPAWN action is allowed. Forcing RESPAWN.")
                action = int(ActionType.RESPAWN)

            # If terminate_on_player_death is True, terminate after respawn action
            if self.terminate_on_player_death:
                observation = self._get_observation()
                return observation, 0.0, True, False, self._get_info()

        # With action masking, invalid actions should not be selected by the model
        # However, we still track them for diagnostics and ENFORCE the masking
        state_valid = True
        state_invalid_reason = ""
        if action_valid:  # Only check state validity if action code is valid
            state_valid, state_invalid_reason = self._is_action_valid_for_state(action, self.player_ship, is_player=True)
            if not state_valid:
                self.state_invalid_action_count += 1
                if self.warn_on_invalid_action:
                    logger.warning(f"Action {ActionType(action).name} invalid for current state: {state_invalid_reason}. "
                                 f"This should not happen with action masking!")

                # ENFORCE action masking: force invalid action to appropriate valid action
                # This prevents the model from executing invalid actions. The
                # rules, fallback priority and enforcement all live in the shared
                # action_masker utility (single source of truth with bot_v6).
                original_action = action
                state = self._build_mask_state(self.player_ship, is_player=True)
                mask = action_masker.get_action_mask(state)
                action = action_masker.mask_action(original_action, state, mask=mask)
                if self.warn_on_invalid_action and action != original_action:
                    logger.warning(f"Forcing {ActionType(original_action).name} -> {ActionType(action).name} (action mask enforcement)")

        reward = 0.0
        self.current_step += 1
        self.action_counter += 1  # Increment action counter (tracks actions taken this episode)

        # Execute player action
        self.last_player_action = int(action) if action is not None else None
        # Provide previous position for reward components that rely on positional delta
        prev_position = (self.player_ship['x'], self.player_ship['y'])
        # Clear just_recharged flag when executing a non-RECHARGE action to prevent recharge loops
        if action != int(ActionType.RECHARGE):
            self.player_ship['just_recharged'] = False
        action_reward, action_info = self._execute_action(action, self.player_ship, is_player=True, target=action_target, energy=action_energy)
        # Per-ship tick order (step 4): drain DRAINING shields after the player's action.
        self._post_action_tick(self.player_ship)
        # Expose previous position so RewardComponents (e.g., DistanceToAsteroidReward) can compute deltas
        action_info['prev_position'] = prev_position

        # Annotate action_info with validation/debug fields
        action_info['requested_action'] = str(requested_action_raw)
        action_info['valid_action'] = bool(action_valid)
        # After enforcement, the executed action is valid even if the original was not
        action_info['state_valid'] = True
        if not state_valid:
            action_info['state_invalid_reason'] = state_invalid_reason
            action_info['state_enforced'] = True  # Flag that enforcement was applied
        # keep raw reward in action_info for transparency/debugging
        action_info['raw_reward'] = float(action_reward)

        # Will compute scaled reward below; store scaled in action_info for renderer
        # compute final reward via RewardCalculator; pass env and ship for optional shaping
        scaled = self.reward_calc.compute(action_reward, action, action_info, env=self, ship=self.player_ship)
        reward += scaled
        # record scaled reward in action_info for rendering/debug
        action_info['scaled_reward'] = float(scaled)
        # Save last player action result for rendering
        try:
            # shallow copy of relevant fields, include optional payload
            self.last_player_action_result = {
                'action': action_info.get('action'),
                'success': action_info.get('success'),
                'raw_reward': float(action_info.get('raw_reward', 0.0)),
                'scaled_reward': float(action_info.get('scaled_reward', 0.0)),
                'state_valid': action_info.get('state_valid', True),
                'state_invalid_reason': action_info.get('state_invalid_reason', ''),
                'payload': action_info.get('payload', None)
            }
        except Exception:
            self.last_player_action_result = None

        # Execute opponent actions (simple AI)
        for i, opponent in enumerate(self.opponent_ships):
            if not opponent['destroyed']:
                # Per-ship tick order (steps 1-2): maintenance + recharge before action.
                self._pre_action_tick(opponent)
                opponent_action = self._get_opponent_action(opponent)
                # Response-bot opponents (BOT_V2/BOT_V3) stash their own target/
                # energy so the simulator executes exactly what the bot asked for
                # (no auto-targeting). Other AIs leave these unset -> None -> the
                # action falls back to its legacy auto-target selection.
                op_target = opponent.pop('_pending_action_target', None)
                op_energy = opponent.pop('_pending_action_energy', None)
                # Clear just_recharged flag for opponents (same as player) to prevent recharge lock
                if opponent_action != int(ActionType.RECHARGE):
                    opponent['just_recharged'] = False
                # store action id
                try:
                    self.last_opponent_actions[i] = int(opponent_action)
                except Exception:
                    self.last_opponent_actions[i] = None
                # execute and capture result
                r_op, info_op = self._execute_action(
                    opponent_action, opponent, is_player=False,
                    target=op_target, energy=op_energy,
                )
                # Per-ship tick order (step 4): drain DRAINING shields after action.
                self._post_action_tick(opponent)
                # store a compact result for rendering, include optional payload
                try:
                    self.last_opponent_action_results[i] = {
                        'action': info_op.get('action'),
                        'success': info_op.get('success'),
                        'raw_reward': float(r_op),
                        'payload': info_op.get('payload', None)
                    }
                except Exception:
                    self.last_opponent_action_results[i] = None
            else:
                self.last_opponent_actions[i] = None
                self.last_opponent_action_results[i] = None

        # Update combat states based on current positions (set COMBAT when ships share a zone)
        try:
            self._update_combat_states()
        except Exception:
            # Non-fatal: if update_combat_states fails for any reason, log and continue
            logger.exception("Failed to update combat states")

        # Check termination conditions
        # Only terminate on player death if flag is set (for training)
        # Otherwise, let the game run to max_steps (for full simulation)
        if self.terminate_on_player_death:
            terminated = self.player_ship['destroyed']
        else:
            terminated = False  # Never terminate early in simulation mode

        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()
        info.update(action_info)

        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up resources"""
        pass


# Register the environment with Gymnasium
gym.register(
    id='ProspectorsPirates-v0',
    entry_point='pnp_env:ProspectorsPiratesEnv',
)
