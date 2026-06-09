"""
Prospectors n Pirates Game Environment
Compatible with OpenAI Gym/Gymnasium for Deep Reinforcement Learning
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import random
import math
import os
import logging
from enum import IntEnum

# Import stable-baselines3 models for enemy AI
try:
    from stable_baselines3 import PPO, DQN, A2C
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = DQN = A2C = None

# Import modular reward utilities from separate module
from reward_utils import RewardComponent, RewardCalculatorComposite, DistanceToAsteroidReward, SurvivalReward

# Module logger
logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Game action types"""
    WAIT = 0
    MINE = 1
    MOVE_NORTH = 2
    MOVE_SOUTH = 3
    MOVE_EAST = 4
    MOVE_WEST = 5
    RECHARGE = 6
    RECHARGE_END = 7
    ATTACK = 8
    JUMP_TO_ASTEROID = 9  # previously JUMP
    SELL = 10
    RAISE_SHIELDS = 11
    JUMP_TO_TRADING_POST = 12  # jump to nearest trading post
    RESPAWN = 13  # respawn after being destroyed


class OpponentAIType(IntEnum):
    """AI behavior types for opponent ships"""
    PROSPECTOR = 0  # Focuses on mining and selling, avoids combat
    PIRATE = 1      # Aggressive, prioritizes attacking and stealing
    HEURISTIC = 2   # Balanced approach with smart decision-making
    MODEL = 3       # Uses a trained RL model for decision-making


@dataclass
class RewardConfig:
    """Configuration for reward shaping and scaling."""
    action_multipliers: Dict[int, float] = field(default_factory=lambda: {
        ActionType.WAIT: 1.0,
        ActionType.MINE: 1.0,
        ActionType.MOVE_NORTH: 1.0,
        ActionType.MOVE_SOUTH: 1.0,
        ActionType.MOVE_EAST: 1.0,
        ActionType.MOVE_WEST: 1.0,
        ActionType.RECHARGE: 1.0,
        ActionType.RECHARGE_END: 1.0,
        ActionType.ATTACK: 1.0,
        ActionType.JUMP_TO_ASTEROID: 1.0,
        ActionType.SELL: 1.0,
        ActionType.RAISE_SHIELDS: 0.0,
    })
    success_bonus: float = 0.0
    failure_penalty: float = 0.0
    # If True, the environment will construct a RewardCalculatorComposite automatically
    use_composite: bool = True
    # Optional list of component specifications to include in the composite.
    # Each entry can be either a string (component class name) or a dict:
    #  - {'name': 'DistanceToAsteroidReward', 'params': {'weight': 0.05}}
    composite_components: Optional[List[object]] = None


class RewardCalculator:
    """Applies RewardConfig to raw action rewards and optional shaping.

    The compute method takes the raw reward returned by action execution and
    returns a final scalar reward that will be accumulated by the environment.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def compute(self, raw_reward: float, action: int, action_info: dict, *, env: Optional[object] = None, ship: Optional[dict] = None) -> float:
        # Defensive cast
        try:
            a = int(action)
        except Exception:
            a = action

        multiplier = self.config.action_multipliers.get(a, 1.0)
        reward = float(raw_reward) * float(multiplier)

        # Apply success/failure adjustments if action_info provides 'success'
        success = action_info.get('success', None)
        if success is True:
            reward += self.config.success_bonus
        elif success is False:
            reward += self.config.failure_penalty

        # Example hook: small shaping based on ship health (optional)
        # If ship provided and health is very low, give small positive shaping to encourage survival actions
        if ship is not None:
            try:
                health_frac = ship.get('health', 0) / max(1.0, getattr(env, 'config', {}).get('max_health', 100))
                if health_frac < 0.2 and a in (ActionType.RECHARGE, ActionType.RECHARGE_END, ActionType.MOVE_NORTH,
                                              ActionType.MOVE_SOUTH, ActionType.MOVE_EAST, ActionType.MOVE_WEST):
                    # small encouragement to take defensive/mobility actions when low HP
                    reward += 0.01
            except Exception:
                pass

        return float(reward)


class ProspectorsPiratesEnv(gym.Env):
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
        """
        super().__init__()

        self.map_width = map_width
        self.map_height = map_height
        self.num_opponents = num_opponents
        self.forced_opponent_types = forced_opponent_types
        if forced_opponent_types:
            self.num_opponents = len(forced_opponent_types)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.terminate_on_player_death = terminate_on_player_death
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
        self._enemy_model_paths = None  # Cache for loaded model paths
        self._enemy_models = {}  # Cache for loaded model instances {path: model_instance}

        # Game configuration (based on game rules)
        self.config = {
            'max_energy': 100,
            'energy_per_recharge': 10,
            'max_health': 100,
            'max_nutrinium_cargo': 1000,
            'max_credits': 10000,
            'max_skill_points': 20,
            'energy_costs': {
                'mine': 5,
                'move': 2,
                'jump': 5,  # per unit distance
                'attack': 1,  # minimum
                'shields': 1,
            },
            'combat': {
                'base_hit_chance': 0.5,
                'base_shield_resistance': 0.25,
                'recharge_penalty': 0.2,
                'damage_variance': 0.5,  # +- 50%
            },
            'mining': {
                'base_success_chance': 0.5,
                'payout_modifier': 1.0,
                'min_payout': 1,
                'max_payout': 10,
            },
            'market': {
                'nutrinium_price': 3,
                'ship_cost': 100,
            },
            'asteroid_density': 0.15,
            'trading_post_count': 4,
            'sensor_range': 5,
            # Nutrinium distribution tuning
            # Total nutrinium budget = nutrinium_per_player * total_players * (1 +/- budget_variance)
            # This ensures enough nutrinium for all players to mine competitively,
            # but not so much that accumulating credits becomes trivial.
            'nutrinium_per_player': 50,         # target nutrinium units available per player
            'nutrinium_budget_variance': 0.2,   # +/-20% random variance on total budget
            # Beta distribution shape parameters for nutrinium concentration per asteroid.
            # alpha < beta -> skew toward lower concentrations (more poor asteroids).
            # With alpha=1.5, beta=8: ~30% poor, ~50% medium, ~20% rich asteroids.
            # The few rich asteroids become high-value strategic targets.
            'nutrinium_beta_alpha': 1.5,
            'nutrinium_beta_beta': 8.0,
            # Mass range for asteroids (larger masses lower average concentration,
            # creating more strategic differentiation between poor and rich asteroids)
            'asteroid_mass_min': 20,
            'asteroid_mass_max': 80,
            # Ship abilities max values for normalization
            'abilities': {
                'energy_max': 10,
                'recharge_energy': 10,
                'mine_accuracy': 10,
                'mine_yield_multiplier': 5,
                'mine_cost': 10,
                'combat_salvage_multiplier': 5,
                'attack_accuracy': 10,
                'attack_power': 10,
                'evade': 10,
                'shield_strength': 10,
                'jump_distance': 10,
            },
            # Observation parameters
            'top_asteroids_count': 5  # Number of top asteroids to include in observation
        }

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

        # Action space: discrete actions
        # 0: WAIT, 1: MINE, 2-5: MOVE (N,S,E,W), 6: RECHARGE, 7: RECHARGE_END,
        # 8: ATTACK (closest enemy), 9: JUMP_TO_ASTEROID (to nearest asteroid), 10: SELL,
        # 11: RAISE_SHIELDS, 12: JUMP_TO_TRADING_POST, 13: RESPAWN
        self.action_space = spaces.Discrete(14)

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
        #   - Strongest enemy (x, y, energy, health, nutrinium, credits, combat_score) = 7 values
        #   - Weakest enemy (x, y, energy, health, nutrinium, credits, combat_score) = 7 values
        # Total enemy info: 14 values

        ship_state_size = 24  # Complete ship state (including action counter)
        strategic_context_size = 8  # High-signal strategic features
        sensor_grid_size = (2 * self.config['sensor_range'] + 1) ** 2
        top_asteroids_size = self.config['top_asteroids_count'] * 6  # 5 asteroids * 6 features each
        trading_post_size = 3  # x, y, distance
        enemy_info_size = 14  # 2 enemies * 7 features each

        obs_size = (
            ship_state_size +
            strategic_context_size +
            sensor_grid_size +
            top_asteroids_size +
            trading_post_size +
            enemy_info_size
        )

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
                shape=(14,),  # One mask value per action
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0

        # Reset action tracking
        self.action_counter = 0  # Reset action counter for new episode
        self.last_player_action = None
        self.last_opponent_actions = {}
        self.last_player_action_result = None
        self.last_opponent_action_results = {}

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

        # Generate shared abilities for this episode - all participants get the same abilities
        self._episode_abilities = self._generate_random_abilities()

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
            'shields_up': False,
            'destroyed': False,
            'state': 'READY',  # READY, RECHARGING, DESTROYED
            'respawn_count': 0,  # Track number of times respawned this episode
            'skill_points_total': 5,
            'skill_points_spent': 0,
            # Ship abilities - same as all other participants this episode
            'abilities': dict(self._episode_abilities),
        }

        # Initialize opponent ships
        self.opponent_ships = []

        # Load available enemy models
        enemy_model_paths = self._load_enemy_model_paths()

        for i in range(self.num_opponents):
            # Decide AI type: use forced list if provided, otherwise random
            use_model = False
            model_path = None
            ai_type = None

            if self.forced_opponent_types and i < len(self.forced_opponent_types):
                # Use forced opponent type from the list
                forced_type = self.forced_opponent_types[i]
                # forced_type is a string: HEURISTIC, PIRATE, PROSPECTOR, or a model path
                forced_upper = forced_type.upper()
                if forced_upper == 'HEURISTIC':
                    ai_type = OpponentAIType.HEURISTIC
                elif forced_upper == 'PIRATE':
                    ai_type = OpponentAIType.PIRATE
                elif forced_upper == 'PROSPECTOR':
                    ai_type = OpponentAIType.PROSPECTOR
                else:
                    # Treat as model path
                    ai_type = OpponentAIType.MODEL
                    use_model = True
                    model_path = forced_type
            else:
                # No forced type - use random selection
                if enemy_model_paths and len(enemy_model_paths) > 0:
                    if random.random() < 0.15:  # 15% chance to use model (reduced from 50% for training performance)
                        use_model = True
                        model_path = random.choice(enemy_model_paths)
                        ai_type = OpponentAIType.MODEL

                if ai_type is None:
                    # Randomly assign algorithm-based AI type with weighted distribution
                    # 40% prospector, 30% pirate, 30% heuristic
                    ai_type_roll = random.random()
                    if ai_type_roll < 0.4:
                        ai_type = OpponentAIType.PROSPECTOR
                    elif ai_type_roll < 0.7:
                        ai_type = OpponentAIType.PIRATE
                    else:
                        ai_type = OpponentAIType.HEURISTIC

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
                'shields_up': False,
                'destroyed': False,
                'state': 'READY',
                'skill_points_total': 5,
                'skill_points_spent': random.randint(0, 5),
                # Ship abilities - same as all other participants this episode
                'abilities': dict(self._episode_abilities),
            }
            self.opponent_ships.append(ship)

        # Generate asteroids
        self.asteroids = []

        if self.use_predefined_asteroids:
            # Load predefined asteroids from config file
            predefined = self._load_predefined_asteroids()
            if predefined is not None:
                # Deep copy the predefined asteroids to avoid modifying cached data
                import copy
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
                        # As a last resort, find any free cell
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

                # --- Nutrinium budget distribution across asteroids ---
                # Step 1: Compute total nutrinium budget relative to player count.
                # Budget = nutrinium_per_player * total_players * (1 +/- variance).
                # This ensures competitive but not trivial nutrinium availability.
                total_players = 1 + self.num_opponents
                base_budget = self.config.get('nutrinium_per_player', 50) * total_players
                variance = self.config.get('nutrinium_budget_variance', 0.2)
                budget = int(base_budget * random.uniform(1.0 - variance, 1.0 + variance))
                budget = max(total_players, budget)  # at least 1 per player

                if self.asteroids:
                    n_ast = len(self.asteroids)
                    alpha = self.config.get('nutrinium_beta_alpha', 1.5)
                    beta_param = self.config.get('nutrinium_beta_beta', 8.0)

                    # Step 2: Draw a target concentration for each asteroid from a
                    # Beta distribution. Beta(1.5, 8) produces a right-skewed
                    # distribution: many poor asteroids, few rich ones.
                    # The concentrations are applied directly to each asteroid's
                    # mass so the richness pattern is preserved regardless of
                    # budget scaling.
                    target_concentrations = [
                        random.betavariate(alpha, beta_param) for _ in range(n_ast)
                    ]

                    # Step 3: Assign raw nutrinium = floor(concentration x mass).
                    # This gives each asteroid its "natural" deposit based on the
                    # drawn concentration.  Rich asteroids (high Beta draw) get a
                    # large share of their mass as nutrinium; poor ones get little.
                    raw_nutr = [
                        int(c * a['mass']) for c, a in zip(target_concentrations, self.asteroids)
                    ]
                    raw_total = sum(raw_nutr)

                    # Step 4: Scale to hit the budget while preserving concentration
                    # ordering and capping each asteroid at its mass.
                    if raw_total > 0 and raw_total != budget:
                        # First pass: proportional scaling
                        scale = budget / raw_total
                        allocated = [max(0, int(r * scale)) for r in raw_nutr]

                        # Clamp each to mass
                        for i in range(n_ast):
                            allocated[i] = min(allocated[i], self.asteroids[i]['mass'])

                        # Second pass: distribute remaining deficit/surplus among
                        # non-capped asteroids, respecting mass ceiling.
                        deficit = budget - sum(allocated)
                        if deficit > 0:
                            # Sort uncapped asteroids by remaining headroom (desc)
                            headroom = [
                                (self.asteroids[i]['mass'] - allocated[i], i)
                                for i in range(n_ast)
                                if allocated[i] < self.asteroids[i]['mass']
                            ]
                            headroom.sort(reverse=True)
                            for _, idx in headroom:
                                give = min(deficit, self.asteroids[idx]['mass'] - allocated[idx])
                                allocated[idx] += give
                                deficit -= give
                                if deficit <= 0:
                                    break
                        elif deficit < 0:
                            # Over budget - trim from richest first
                            surplus = -deficit
                            richest = sorted(range(n_ast), key=lambda i: allocated[i], reverse=True)
                            for idx in richest:
                                take = min(surplus, allocated[idx])
                                allocated[idx] -= take
                                surplus -= take
                                if surplus <= 0:
                                    break
                    elif raw_total == 0:
                        # All concentrations were ~0; distribute budget evenly
                        per = budget // n_ast
                        allocated = [min(per, a['mass']) for a in self.asteroids]
                        leftover = budget - sum(allocated)
                        for i in range(n_ast):
                            if leftover <= 0:
                                break
                            give = min(leftover, self.asteroids[i]['mass'] - allocated[i])
                            allocated[i] += give
                            leftover -= give
                    else:
                        allocated = raw_nutr

                    # Step 5: Final assignment
                    for i, asteroid in enumerate(self.asteroids):
                        asteroid['nutrinium'] = min(allocated[i], asteroid['mass'])

        # Generate trading posts
        self.trading_posts = []

        if self.use_predefined_asteroids:
            # Load predefined trading posts from the same config file
            predefined_posts = self._load_predefined_trading_posts()
            if predefined_posts is not None:
                # Deep copy the predefined trading posts
                import copy
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
        needed = self.config.get('trading_post_count', 0) - len(self.trading_posts)
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
                        # Fallback: find any available cell
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

        # Build spatial lookup cache for fast entity-at-location queries
        self._rebuild_location_cache()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

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
                logger.warning(f"  Falling back to random asteroid generation.")
                return None

            with open(self.asteroid_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if dimension_key not in config:
                logger.warning(f"No asteroid configuration found for dimension '{dimension_key}' in {self.asteroid_config_path}")
                logger.warning(f"  Available dimensions: {list(config.keys())}")
                logger.warning(f"  Falling back to random asteroid generation.")
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
                    logger.warning(f"Invalid asteroid data at index {i}: {asteroid_data}")
                    continue

                # Validate coordinates
                ax = int(asteroid_data['x'])
                ay = int(asteroid_data['y'])
                if not (0 <= ax < self.map_width and 0 <= ay < self.map_height):
                    logger.warning(f"Asteroid at index {i} has out-of-bounds coordinates: ({asteroid_data['x']}, {asteroid_data['y']})")
                    continue

                # Ensure nutrinium doesn't exceed mass
                mass = int(asteroid_data['mass'])
                nutrinium = int(min(asteroid_data['nutrinium'], mass))

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

    def _load_predefined_trading_posts(self) -> Optional[List[dict]]:
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

    def _load_predefined_start_position(self) -> Optional[dict]:
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
                logger.warning(f"  Falling back to random starting position.")
                return None

            with open(self.start_position_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if dimension_key not in config:
                logger.warning(f"No start position configuration found for dimension '{dimension_key}' in {self.start_position_config_path}")
                logger.warning(f"  Available dimensions: {list(config.keys())}")
                logger.warning(f"  Falling back to random starting position.")
                return None

            start_data = config[dimension_key]

            # Validate start position data
            if not all(k in start_data for k in ['player_x', 'player_y']):
                logger.warning(f"Invalid start position data (missing player_x or player_y): {start_data}")
                logger.warning(f"  Falling back to random starting position.")
                return None

            # Validate coordinates
            if not (0 <= start_data['player_x'] < self.map_width and 0 <= start_data['player_y'] < self.map_height):
                logger.warning(f"Start position out of bounds: ({start_data['player_x']}, {start_data['player_y']})")
                logger.warning(f"  Falling back to random starting position.")
                return None

            start_position = {
                'player_x': int(start_data['player_x']),
                'player_y': int(start_data['player_y'])
            }

            # Cache the loaded data
            self._predefined_start_cache = start_position
            logger.info(f"Loaded predefined start position from {self.start_position_config_path} for dimension {dimension_key}: ({start_position['player_x']}, {start_position['player_y']})")

            return start_position

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in start position config file {self.start_position_config_path}: {e}")
            logger.warning(f"  Falling back to random starting position.")
            return None
        except Exception as e:
            logger.error(f"Error loading start position config: {e}")
            logger.warning(f"  Falling back to random starting position.")
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
        # Use cached data if available
        if self._enemy_model_paths is not None:
            return self._enemy_model_paths

        try:
            if not os.path.exists(self.enemy_models_config_path):
                logger.info(f"Enemy models config file not found: {self.enemy_models_config_path}")
                logger.info(f"  No model-based enemies will be created.")
                return None

            model_paths = []
            with open(self.enemy_models_config_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Strip whitespace
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Add model path
                    model_paths.append(line)

            if not model_paths:
                logger.info(f"No model paths found in {self.enemy_models_config_path}")
                logger.info(f"  No model-based enemies will be created.")
                return None

            # Cache the loaded paths
            self._enemy_model_paths = model_paths
            logger.info(f"Loaded {len(model_paths)} enemy model paths from {self.enemy_models_config_path}")

            return model_paths

        except Exception as e:
            logger.warning(f"Error loading enemy model paths: {e}")
            return None

    def _generate_random_abilities(self) -> dict:
        """Generate a random set of abilities for this episode.
        All participants (player and enemies) share the same abilities per episode."""
        return {
            'energy_max': random.randint(0, 5),
            'recharge_energy': random.randint(0, 3),
            'mine_accuracy': random.randint(0, 3),
            'mine_yield_multiplier': random.randint(1, 2),
            'mine_cost': random.randint(1, 3),
            'combat_salvage_multiplier': random.randint(0, 2),
            'sensor_range': random.randint(1, 3),
            'attack_accuracy': random.randint(0, 5),
            'attack_power': random.randint(0, 5),
            'evade': random.randint(0, 3),
            'shield_strength': random.randint(0, 3),
            'jump_distance': random.randint(0, 3),
        }

    def _load_enemy_model(self, model_path: str):
        """
        Load a trained model for enemy AI.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model instance, or None if loading fails
        """
        # Check cache first
        if model_path in self._enemy_models:
            return self._enemy_models[model_path]

        # Check if stable-baselines3 is available
        if not SB3_AVAILABLE:
            logger.warning(f"stable-baselines3 not available, cannot load enemy model: {model_path}")
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
                logger.warning(f"Failed to load enemy model: {model_path}")
                return None

            # Check action space compatibility
            model_action_space = temp_model.action_space.n
            env_action_space = self.action_space.n

            if model_action_space != env_action_space:
                # Check if this is the known 15 vs 14 issue (LOWER_SHIELDS removed)
                if model_action_space == 15 and env_action_space == 14:
                    # Old model (15 actions with LOWER_SHIELDS) vs new environment (14 actions)
                    # Actions 0-11 are the same, 12 (LOWER_SHIELDS) -> WAIT, 13->12, 14->13
                    logger.info(f"Enemy model {model_path} has old action space (15), using compatibility mode")
                else:
                    logger.warning(f"Enemy model {model_path} has incompatible action space: {model_action_space} vs {env_action_space}")
                    return None

            # Reload model WITH environment binding for proper normalization and observation processing
            # This ensures the enemy model gets the same treatment as the player model,
            # preventing performance gaps due to different observation normalization.
            # Skip env binding if observation spaces are incompatible (e.g. old model with 128 obs vs 200).
            obs_compat = True
            try:
                from gymnasium import spaces as _spaces
                model_obs_space = temp_model.observation_space
                if (isinstance(model_obs_space, _spaces.Dict) and
                        isinstance(self.observation_space, _spaces.Dict) and
                        'observation' in model_obs_space.spaces and
                        'observation' in self.observation_space.spaces):
                    if model_obs_space['observation'].shape != self.observation_space['observation'].shape:
                        obs_compat = False
                        logger.info(f"Enemy model {model_path} has different obs size "
                                    f"({model_obs_space['observation'].shape[0]} vs "
                                    f"{self.observation_space['observation'].shape[0]}), "
                                    f"skipping env binding")
            except Exception:
                pass

            if not obs_compat:
                self._enemy_models[model_path] = temp_model
                logger.info(f"Loaded enemy model without env binding: {model_path} (action space: {model_action_space})")
                return temp_model

            try:
                # Determine which algorithm to use for reload
                final_model = None
                if 'ppo' in model_path.lower() or isinstance(temp_model, PPO):
                    final_model = PPO.load(model_path, env=self)
                elif 'dqn' in model_path.lower() or isinstance(temp_model, DQN):
                    final_model = DQN.load(model_path, env=self)
                else:
                    final_model = A2C.load(model_path, env=self)

                # Cache the environment-bound model
                self._enemy_models[model_path] = final_model
                logger.info(f"Loaded enemy model with env binding: {model_path} (action space: {model_action_space})")
                return final_model
            except Exception as e:
                # Fallback to model without env binding if reload fails
                logger.warning(f"Failed to reload model with env binding ({e}), using original model without binding")
                self._enemy_models[model_path] = temp_model
                logger.info(f"Loaded enemy model without env binding: {model_path} (action space: {model_action_space})")
                return temp_model

        except Exception as e:
            logger.warning(f"Error loading enemy model {model_path}: {e}")
            return None

    def _normalize_action(self, action) -> int:
        """Normalize various action types (numpy array, list, tuple, scalar) into int."""
        # Numpy array
        try:
            if isinstance(action, np.ndarray):
                try:
                    return int(action.item())
                except Exception:
                    # Fallback to first element
                    return int(action[0])
            # Lists or tuples
            if isinstance(action, (list, tuple)):
                return int(action[0])
            # Scalars (int-like)
            return int(action)
        except Exception as e:
            raise ValueError(f"Unable to normalize action to int: {action!r}") from e

    def _is_action_valid_for_state(self, action: int, ship: dict, is_player: bool = True) -> Tuple[bool, str]:
        """
        Validate if an action is valid given the current game state.

        Enhanced action masking rules:
        1. DESTROYED state: only RESPAWN is valid
        2. Not DESTROYED: RESPAWN is invalid
        3. RECHARGING state: only WAIT and RECHARGE_END are valid
        4. Not RECHARGING: RECHARGE_END is invalid (WAIT is always valid for gaining action points)
        5. RECHARGING + full energy: only RECHARGE_END is valid
        6. ATTACK: requires enemy in same zone
        7. MINE: requires asteroid at current location
        8. SELL: requires trading post at current location
        9. All actions: respect energy requirements
        10. Energy-consuming actions masked when insufficient energy
        11. JUMP_TO_TRADING_POST and SELL: require nutrinium
        12. RAISE_SHIELDS: requires combat situation (enemy in same zone)
        13. JUMP_TO_ASTEROID: masked when already at asteroid with nutrinium >= 5%
        14. JUMP_TO_ASTEROID: masked when nearest asteroid is at same location (distance 0, would be a no-op)
        15. RECHARGE: masked when energy > 50% (avoid wasteful recharge cycles)
        16. JUMP_TO_TRADING_POST: masked when already at a trading post (use SELL)
        17. WAIT: masked when energy is critically low (< min useful cost) and NOT recharging
            (prevents dead-end: WAIT doesn't restore energy, only RECHARGE does)

        Args:
            action: The action to validate
            ship: The ship attempting the action
            is_player: Whether this is the player ship

        Returns:
            (is_valid, reason) - True if valid, False with reason string if invalid
        """
        # Rule 1: If DESTROYED, only RESPAWN is valid
        if ship.get('destroyed', False):
            if action == ActionType.RESPAWN:
                return True, ""
            else:
                return False, "ship is destroyed, only RESPAWN is valid"

        # Rule 2: If NOT destroyed, RESPAWN is invalid
        if action == ActionType.RESPAWN:
            return False, "can only respawn when destroyed"

        # Rule 3 & 5: If RECHARGING, only WAIT and RECHARGE_END are valid
        # If recharging + full energy, only RECHARGE_END is valid
        # Rule 3b: RECHARGE_END is masked until energy >= 50% to prevent
        #          inefficient short recharge cycles (e.g. recharging from 7->17
        #          then immediately jumping/mining back to low energy)
        if ship.get('recharging', False):
            recharge_end_threshold = int(self.config['max_energy'] * 0.5)
            if ship['energy'] >= self.config['max_energy']:
                # Rule 5: Full energy while recharging -> only RECHARGE_END
                if action == ActionType.RECHARGE_END:
                    return True, ""
                else:
                    return False, "energy full while recharging, must end recharge"
            elif ship['energy'] < recharge_end_threshold:
                # Rule 3b: Energy too low to end recharging -> only WAIT
                if action == ActionType.WAIT:
                    return True, ""
                else:
                    return False, f"energy too low to end recharge ({ship['energy']}/{self.config['max_energy']}, need {recharge_end_threshold})"
            else:
                # Rule 3: Recharging with sufficient energy -> WAIT or RECHARGE_END
                if action == ActionType.WAIT:
                    return True, ""
                elif action == ActionType.RECHARGE_END:
                    return True, ""
                else:
                    return False, "while recharging, can only WAIT or RECHARGE_END"

        # Rule 4: If NOT recharging, RECHARGE_END is invalid
        # Note: WAIT is always valid (ship does nothing / gains action points)
        if action == ActionType.RECHARGE_END:
            return False, "not currently recharging"

        # From here: ship is not destroyed and not recharging
        # Validate specific actions with their requirements

        # WAIT - generally valid but masked when energy is critically low and not recharging.
        # Per game rules: "WAIT does not require ENERGY. If the ship is RECHARGING
        # then it generates ENERGY." So WAIT without recharging at low energy is a
        # dead-end that traps the agent forever (can't do anything useful, energy
        # never recovers). Force the agent to RECHARGE instead.
        if action == ActionType.WAIT:
            if not ship.get('recharging', False):
                # Check if energy is too low to perform any useful action
                # Move (2 energy) is the cheapest broadly-useful action.
                # Attack (1 energy) requires enemy in same zone (rare/situational).
                # If energy < move cost, the player can't do anything productive
                # and must RECHARGE to recover energy.
                min_useful_energy = self.config['energy_costs']['move']
                if ship['energy'] < min_useful_energy:
                    return False, "energy too low to do anything useful, must RECHARGE instead of WAIT"
            return True, ""

        # MINE - requires asteroid at current location with nutrinium
        if action == ActionType.MINE:
            # Rule 10: Check energy requirement
            if ship['energy'] < self.config['energy_costs']['mine']:
                return False, "insufficient energy to mine"
            # Rule 8: Check asteroid at location
            asteroid = self._get_entity_at_location(ship['x'], ship['y'], self.asteroids)
            if asteroid is None:
                return False, "no asteroid at current location"
            if asteroid['nutrinium'] <= 0:
                return False, "asteroid has no nutrinium"
            return True, ""

        # MOVE actions - require sufficient energy (Rule 10)
        if action in [ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH, ActionType.MOVE_EAST, ActionType.MOVE_WEST]:
            if ship['energy'] < self.config['energy_costs']['move']:
                return False, "insufficient energy to move"
            # Check if move would go off map
            new_x, new_y = ship['x'], ship['y']
            if action == ActionType.MOVE_NORTH:
                new_y = ship['y'] - 1
            elif action == ActionType.MOVE_SOUTH:
                new_y = ship['y'] + 1
            elif action == ActionType.MOVE_EAST:
                new_x = ship['x'] + 1
            elif action == ActionType.MOVE_WEST:
                new_x = ship['x'] - 1

            if new_x < 0 or new_x >= self.map_width or new_y < 0 or new_y >= self.map_height:
                return False, "would move off map"
            return True, ""

        # RECHARGE - can only start if not already recharging and energy is low enough
        # Masked when energy > 50% to prevent wasteful recharge cycles
        # Also masked immediately after ending a recharge to prevent recharge loops
        if action == ActionType.RECHARGE:
            # Already checked: not recharging (otherwise would have been caught above)
            if ship.get('just_recharged', False):
                return False, "just finished recharging, do something productive first"
            if ship['energy'] >= self.config['max_energy']:
                return False, "energy already full"
            recharge_threshold = int(self.config['max_energy'] * 0.3)
            if ship['energy'] > recharge_threshold:
                return False, f"energy too high to recharge ({ship['energy']}/{self.config['max_energy']}, threshold {recharge_threshold})"
            return True, ""

        # RECHARGE_END - already handled in recharging state check above
        # This code path won't be reached for RECHARGE_END

        # ATTACK - requires enemy in same zone and sufficient energy
        if action == ActionType.ATTACK:
            # Rule 10: Check energy requirement
            if ship['energy'] < self.config['energy_costs']['attack']:
                return False, "insufficient energy to attack"

            # Rule 7: Check if any other ship exists in the same zone
            if is_player:
                targets = self.opponent_ships
            else:
                # Opponents can target the player OR other opponents in the same zone
                targets = [self.player_ship] + [s for s in self.opponent_ships if s is not ship]

            active_targets = [t for t in targets if not t.get('destroyed', False)]

            if not active_targets:
                return False, "no enemy ships available"

            # ATTACK requires an enemy in the same zone (same x,y)
            enemy_in_zone = any(
                t['x'] == ship['x'] and t['y'] == ship['y']
                for t in active_targets
            )
            if not enemy_in_zone:
                return False, "no enemy in same zone"

            return True, ""

        # JUMP_TO_ASTEROID - requires asteroids and sufficient energy (Rule 10)
        # Masked when already at an asteroid with nutrinium concentration >= 5%
        # Masked when the nearest asteroid is at the same location (distance 0, would do nothing)
        if action == ActionType.JUMP_TO_ASTEROID:
            # Check if already at a mineable asteroid with sufficient nutrinium
            current_asteroid = self._get_entity_at_location(ship['x'], ship['y'], self.asteroids)
            if current_asteroid is not None and current_asteroid.get('nutrinium', 0) > 0:
                mass = max(current_asteroid.get('mass', 1), 1)
                concentration = current_asteroid['nutrinium'] / mass
                if concentration >= 0.05:  # 5% threshold
                    return False, f"already at asteroid with {concentration:.0%} nutrinium, mine it first"

            # Find best asteroid by score (same logic as _action_jump)
            top = self._get_top_asteroids(ship['x'], ship['y'], count=1)
            if not top:
                return False, "no asteroids available"

            best = top[0]
            distance = best['distance']

            # If the best asteroid is at the same location (distance 0), the jump
            # would be a no-op. Mask it to prevent infinite loops.
            if distance == 0:
                return False, "best asteroid is at current location (distance 0), mine it or move away"

            energy_cost = int(distance * self.config['energy_costs']['jump'])

            if ship['energy'] < energy_cost:
                return False, f"insufficient energy (need {energy_cost}, have {ship['energy']})"

            return True, ""

        # JUMP_TO_TRADING_POST - requires trading posts, sufficient energy, and nutrinium (Rule 11)
        # Masked when already at a trading post (should SELL instead)
        if action == ActionType.JUMP_TO_TRADING_POST:
            # Rule 11: Requires nutrinium (no point jumping to trading post without anything to sell)
            if ship['nutrinium'] < 10:
                return False, "not enough nutrinium to justify jumping to trading post (need >= 10)"

            # Already at a trading post -- just SELL instead
            current_post = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
            if current_post is not None:
                return False, "already at a trading post, use SELL instead"

            post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
            if post is None:
                return False, "no trading posts available"

            distance = self._calculate_distance(ship['x'], ship['y'], post['x'], post['y'])
            energy_cost = int(distance * self.config['energy_costs']['jump'])

            if ship['energy'] < energy_cost:
                return False, f"insufficient energy (need {energy_cost}, have {ship['energy']})"

            return True, ""

        # SELL - requires being at trading post with nutrinium (Rules 9 & 11)
        if action == ActionType.SELL:
            trading_post = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
            if trading_post is None:
                return False, "not at a trading post"
            # Rule 11: Requires nutrinium
            if ship['nutrinium'] <= 0:
                return False, "no nutrinium to sell"
            return True, ""

        # RAISE_SHIELDS - requires shields down, sufficient energy, and enemy threat in same zone (Rules 10 & 12)
        if action == ActionType.RAISE_SHIELDS:
            if ship['shields_up']:
                reason = "shields already up"
                if self.warn_on_invalid_action or os.getenv('PNP_DEBUG_MASK'):
                    logger.debug(f"RAISE_SHIELDS check: ship={ship.get('name')} shields_up=True -> invalid: {reason}")
                return False, "shields already up"
            if ship['energy'] < self.config['energy_costs']['shields']:
                reason = "insufficient energy for shields"
                if self.warn_on_invalid_action or os.getenv('PNP_DEBUG_MASK'):
                    logger.debug(f"RAISE_SHIELDS check: ship={ship.get('name')} energy={ship.get('energy')} cost={self.config['energy_costs']['shields']} -> invalid: {reason}")
                return False, "insufficient energy for shields"

            # Rule 12 (updated): RAISE_SHIELDS is only valid when there is an enemy threat in the same zone
            # This allows preemptive shield raising when an enemy is detected nearby
            if is_player:
                targets = self.opponent_ships
            else:
                targets = [self.player_ship] + [s for s in self.opponent_ships if s is not ship]
            active_targets = [t for t in targets if not t.get('destroyed', False)]
            if not active_targets:
                reason = "no enemy threat, shields not needed"
                if self.warn_on_invalid_action or os.getenv('PNP_DEBUG_MASK'):
                    logger.debug(f"RAISE_SHIELDS check: ship={ship.get('name')} no active_targets -> invalid: {reason}")
                return False, "no enemy threat, shields not needed"

            enemy_in_same_zone = any((t['x'] == ship['x'] and t['y'] == ship['y']) for t in active_targets)
            if not enemy_in_same_zone:
                reason = "no enemy in same zone, no threat"
                if self.warn_on_invalid_action or os.getenv('PNP_DEBUG_MASK'):
                    enemy_positions = [(t.get('name', '?'), t.get('x'), t.get('y')) for t in active_targets]
                    logger.debug(f"RAISE_SHIELDS check: ship={ship.get('name')} pos=({ship.get('x')},{ship.get('y')}) enemies={enemy_positions} -> invalid: {reason}")
                return False, "no enemy in same zone, no threat"

            if self.warn_on_invalid_action or os.getenv('PNP_DEBUG_MASK'):
                logger.debug(f"RAISE_SHIELDS check: ship={ship.get('name')} pos=({ship.get('x')},{ship.get('y')}) has enemy in zone -> ALLOWED")
            return True, ""

        # Unknown action
        return False, f"unknown action {action}"

    def _get_action_mask(self, ship: dict = None, is_player: bool = True) -> np.ndarray:
        """
        Generate action mask for valid actions given the current game state.

        Args:
            ship: The ship to generate mask for (defaults to player_ship)
            is_player: Whether this ship is the player (affects target selection for ATTACK etc.)

        Returns:
            Boolean array where True means action is valid, False means invalid
        """
        if ship is None:
            ship = self.player_ship

        mask = np.zeros(self.action_space.n, dtype=np.int8)

        # Check each action
        for action in range(self.action_space.n):
            is_valid, _ = self._is_action_valid_for_state(action, ship, is_player=is_player)
            mask[action] = 1 if is_valid else 0

        return mask

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, dict]:
        """Execute one step in the environment"""
        # Rebuild spatial lookup cache for fast entity-at-location queries
        self._rebuild_location_cache()
        # Preserve the raw requested action for debugging
        requested_action_raw = action

        # Normalize action to int (handles numpy arrays, lists, tuples, scalars)
        try:
            action = self._normalize_action(action)
            action_valid = 0 <= action < self.action_space.n
        except Exception as e:
            # Normalization failed (e.g., non-int-like input). Default to WAIT
            self.invalid_action_count += 1
            if self.warn_on_invalid_action:
                logger.warning(f"Unable to normalize action {requested_action_raw!r}: {e}. Defaulting to WAIT.")
            action = int(ActionType.WAIT)
            action_valid = False

        # Validate bounds; if out of range, default to WAIT
        if not (0 <= action < self.action_space.n):
            self.invalid_action_count += 1
            if self.warn_on_invalid_action:
                logger.warning(f"Action {action} out of bounds [0, {self.action_space.n - 1}]. Defaulting to WAIT.")
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
                # This prevents the model from executing invalid actions
                original_action = action

                # Determine the appropriate fallback action based on current state
                if self.player_ship.get('recharging', False):
                    # If recharging with full energy, force to RECHARGE_END
                    if self.player_ship['energy'] >= self.config['max_energy']:
                        action = int(ActionType.RECHARGE_END)
                        if self.warn_on_invalid_action:
                            logger.warning(f"Forcing {ActionType(original_action).name} -> RECHARGE_END (energy full while recharging)")
                    elif original_action not in (int(ActionType.WAIT), int(ActionType.RECHARGE_END)):
                        # Model wants to do something active (ATTACK, MINE, MOVE, etc.)
                        # End recharging so the player can act on the next step
                        action = int(ActionType.RECHARGE_END)
                        if self.warn_on_invalid_action:
                            logger.warning(f"Forcing {ActionType(original_action).name} -> RECHARGE_END (model wants active action, ending recharge)")
                    else:
                        action = int(ActionType.WAIT)
                        if self.warn_on_invalid_action:
                            logger.warning(f"Forcing {ActionType(original_action).name} -> WAIT (recharging)")
                elif self.player_ship.get('destroyed', False):
                    action = int(ActionType.RESPAWN)
                    if self.warn_on_invalid_action:
                        logger.warning(f"Forcing {ActionType(original_action).name} -> RESPAWN (destroyed)")
                else:
                    # Not recharging, not destroyed: pick the best valid action
                    # from the action mask so the player doesn't get stuck on WAIT
                    mask = self._get_action_mask(self.player_ship)
                    # When energy is very low, prioritize RECHARGE to avoid getting stuck
                    if self.player_ship['energy'] <= self.config['energy_costs'].get('move', 5):
                        preferred_fallback_order = [
                            ActionType.RECHARGE,
                            ActionType.MINE,
                            ActionType.SELL,
                            ActionType.WAIT,  # WAIT is free
                            ActionType.JUMP_TO_ASTEROID,
                            ActionType.JUMP_TO_TRADING_POST,
                            ActionType.MOVE_NORTH,
                            ActionType.MOVE_SOUTH,
                            ActionType.MOVE_EAST,
                            ActionType.MOVE_WEST,
                            ActionType.ATTACK,
                            ActionType.RAISE_SHIELDS,
                        ]
                    else:
                        # Prefer productive actions over idle WAIT
                        preferred_fallback_order = [
                            ActionType.MINE,
                            ActionType.SELL,
                            ActionType.JUMP_TO_ASTEROID,
                            ActionType.JUMP_TO_TRADING_POST,
                            ActionType.MOVE_NORTH,
                            ActionType.MOVE_SOUTH,
                            ActionType.MOVE_EAST,
                            ActionType.MOVE_WEST,
                            ActionType.RECHARGE,
                            ActionType.ATTACK,
                            ActionType.RAISE_SHIELDS,
                            ActionType.WAIT,  # last resort
                        ]
                    fallback = int(ActionType.WAIT)
                    for fb_action in preferred_fallback_order:
                        if mask[int(fb_action)] == 1:
                            fallback = int(fb_action)
                            break
                    action = fallback
                    if self.warn_on_invalid_action:
                        logger.warning(f"Forcing {ActionType(original_action).name} -> {ActionType(action).name} (best valid fallback)")

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
        action_reward, action_info = self._execute_action(action, self.player_ship, is_player=True)
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
                opponent_action = self._get_opponent_action(opponent)
                # Clear just_recharged flag for opponents (same as player) to prevent recharge lock
                if opponent_action != int(ActionType.RECHARGE):
                    opponent['just_recharged'] = False
                # store action id
                try:
                    self.last_opponent_actions[i] = int(opponent_action)
                except Exception:
                    self.last_opponent_actions[i] = None
                # execute and capture result
                r_op, info_op = self._execute_action(opponent_action, opponent, is_player=False)
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

        # Update passive effects (recharging)
        self._update_passive_effects()

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

    def _execute_action(self, action: int, ship: dict, is_player: bool = True) -> Tuple[float, dict]:
        """Execute an action for a ship"""
        reward = 0.0
        info = {'action': ActionType(action).name, 'success': False}

        if action == ActionType.WAIT:
            info['success'] = True

        elif action == ActionType.MINE:
            result = self._action_mine(ship)
            # support (reward, success) and (reward, success, payload)
            if isinstance(result, tuple) and len(result) == 3:
                r, success, payload = result
                info['payload'] = payload
            else:
                r, success = result
            reward += r
            info['success'] = success

        elif action in [ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                       ActionType.MOVE_EAST, ActionType.MOVE_WEST]:
            result = self._action_move(ship, action)
            if isinstance(result, tuple) and len(result) == 3:
                r, success, payload = result
                info['payload'] = payload
            else:
                r, success = result
            reward += r
            info['success'] = success

        elif action == ActionType.RECHARGE:
            success = self._action_recharge(ship)
            info['success'] = success

        elif action == ActionType.RECHARGE_END:
            success = self._action_recharge_end(ship)
            info['success'] = success

        elif action == ActionType.RAISE_SHIELDS:
            success = self._action_raise_shields(ship)
            info['success'] = success

        elif action == ActionType.ATTACK:
            result = self._action_attack(ship, is_player)
            if isinstance(result, tuple) and len(result) == 3:
                r, success, payload = result
                info['payload'] = payload
            else:
                r, success = result
            reward += r
            info['success'] = success

        elif action == ActionType.JUMP_TO_ASTEROID:
            result = self._action_jump(ship)
            if isinstance(result, tuple) and len(result) == 3:
                r, success, payload = result
                info['payload'] = payload
            else:
                r, success = result
            reward += r
            info['success'] = success

        elif action == ActionType.JUMP_TO_TRADING_POST:
            result = self._action_jump_to_trading_post(ship)
            if isinstance(result, tuple) and len(result) == 3:
                r, success, payload = result
                info['payload'] = payload
            else:
                r, success = result
            reward += r
            info['success'] = success

        elif action == ActionType.SELL:
            result = self._action_sell(ship)
            if isinstance(result, tuple) and len(result) == 3:
                r, success, payload = result
                info['payload'] = payload
            else:
                r, success = result
            reward += r
            info['success'] = success

        elif action == ActionType.RESPAWN:
            # Store state before respawn to calculate cost
            credits_before = ship.get('credits', 0)
            respawn_count_before = ship.get('respawn_count', 0)

            success = self._action_respawn(ship)
            info['success'] = success

            if success:
                # Calculate actual cost paid
                credits_after = ship.get('credits', 0)
                credits_paid = credits_before - credits_after

                # Calculate what the cost should have been
                base_ship_cost = self.config['market']['ship_cost']
                respawn_cost = base_ship_cost * (respawn_count_before + 1)
                insurance_covered = respawn_cost - credits_paid

                # Reward is negative based on credits lost (not fixed penalty)
                # Scale by a small factor to not overwhelm other rewards
                reward -= credits_paid * 0.1  # 10% of credits lost as negative reward

                # Add payload with respawn details
                info['payload'] = {
                    'respawn_cost': respawn_cost,
                    'credits_paid': credits_paid,
                    'insurance_covered': insurance_covered,
                    'respawn_count': ship.get('respawn_count', 0)
                }

        return reward, info

    def _action_mine(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Mine an asteroid at the current location"""
        if ship['recharging']:
            return -0.1, False

        if ship['energy'] < self.config['energy_costs']['mine']:
            return -0.1, False

        # Find asteroid at current location
        asteroid = self._get_entity_at_location(ship['x'], ship['y'], self.asteroids)
        if asteroid is None:
            return -0.1, False

        # Capture pre-mine state for detailed reporting
        abilities = ship.get('abilities', {})
        energy_before = ship['energy']
        asteroid_mass_before = asteroid['mass']
        asteroid_nutr_before = asteroid['nutrinium']

        ship['energy'] -= self.config['energy_costs']['mine']

        # Calculate mining success based on nutrinium density
        density = asteroid_nutr_before / max(asteroid_mass_before, 1)
        success_chance = min(1.0, density * 10 * self.config['mining']['base_success_chance'])

        # Build detailed mining payload
        mine_details = {
            'asteroid_x': asteroid['x'],
            'asteroid_y': asteroid['y'],
            # Asteroid state before mining
            'ast_mass': f"{asteroid_mass_before}",
            'ast_nutr': f"{asteroid_nutr_before}",
            'ast_density': round(density, 4),
            'success_chance': round(success_chance * 100, 1),
            # Miner skills
            'mine_accuracy': abilities.get('mine_accuracy', 0),
            'mine_yield': abilities.get('mine_yield_multiplier', 1),
            'mine_cost_skill': abilities.get('mine_cost', 2),
            # Energy
            'energy': f"{energy_before}->{ship['energy']}",
            'energy_cost': self.config['energy_costs']['mine'],
        }

        if random.random() < success_chance:
            # Successful mining
            payout = random.randint(
                self.config['mining']['min_payout'],
                min(self.config['mining']['max_payout'], asteroid['nutrinium'])
            )
            asteroid['nutrinium'] -= payout
            asteroid['mass'] -= payout
            ship['nutrinium'] += payout

            mine_details['payout'] = payout
            mine_details['ast_mass_after'] = asteroid['mass']
            mine_details['ast_nutr_after'] = asteroid['nutrinium']
            mine_details['ship_nutr'] = ship['nutrinium']

            return payout * 0.05, True, mine_details  # Reward for mining (precursor to SELL for credits)
        else:
            # Failed mining
            asteroid['mass'] -= 1

            mine_details['payout'] = 0
            mine_details['ast_mass_after'] = asteroid['mass']
            mine_details['ast_nutr_after'] = asteroid['nutrinium']

            return -0.05, False, mine_details

    def _action_move(self, ship: dict, action: int) -> Tuple[float, bool, Optional[dict]]:
        """Move ship in a direction"""
        if ship['energy'] < self.config['energy_costs']['move']:
            return -0.1, False

        new_x, new_y = ship['x'], ship['y']

        if action == ActionType.MOVE_NORTH:
            new_y = max(0, ship['y'] - 1)
        elif action == ActionType.MOVE_SOUTH:
            new_y = min(self.map_height - 1, ship['y'] + 1)
        elif action == ActionType.MOVE_EAST:
            new_x = min(self.map_width - 1, ship['x'] + 1)
        elif action == ActionType.MOVE_WEST:
            new_x = max(0, ship['x'] - 1)

        if new_x == ship['x'] and new_y == ship['y']:
            return -0.1, False, {'from': (ship['x'], ship['y']), 'to': (new_x, new_y)}  # Tried to move off map

        ship['x'] = new_x
        ship['y'] = new_y
        ship['energy'] -= self.config['energy_costs']['move']

        return -0.01, True, {'from': (ship['x'] - (1 if action == ActionType.MOVE_EAST else -1 if action == ActionType.MOVE_WEST else 0), ship['y'] - (1 if action == ActionType.MOVE_SOUTH else -1 if action == ActionType.MOVE_NORTH else 0)), 'to': (ship['x'], ship['y'])}

    def _action_recharge(self, ship: dict) -> bool:
        """Start recharging"""
        if ship['recharging']:
            return False
        ship['recharging'] = True
        return True

    def _action_recharge_end(self, ship: dict) -> bool:
        """Stop recharging"""
        if not ship['recharging']:
            return False
        ship['recharging'] = False
        ship['just_recharged'] = True  # Prevent immediate re-recharging
        return True

    def _action_raise_shields(self, ship: dict) -> bool:
        """Raise shields for a ship.

        Returns True on success, False otherwise.
        - Requires shields to be currently down
        - Requires sufficient energy to power shields (uses config['energy_costs']['shields'])
        - Deducts energy cost when raising shields
        """
        # Already up
        if ship.get('shields_up', False):
            return False

        # Need enough energy to raise shields
        cost = self.config.get('energy_costs', {}).get('shields', 1)
        if ship.get('energy', 0) < cost:
            return False

        ship['shields_up'] = True
        ship['energy'] = max(0, ship.get('energy', 0) - cost)

        # Mark ship as in combat at its own position (shields are same-zone only)
        ship['in_combat'] = True
        ship.setdefault('combat_opponent_positions', set()).add((ship['x'], ship['y']))

        return True

    def _action_jump(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Jump to the best asteroid (by nutrinium-to-distance score).

        Uses the same scoring as _get_top_asteroids so the model's observation
        of top asteroids aligns with where JUMP actually goes.
        """
        if ship['recharging']:
            return -0.1, False, None

        # Find best asteroid by score (nutrinium value vs distance), not just nearest
        top = self._get_top_asteroids(ship['x'], ship['y'], count=1)
        if not top:
            return -0.1, False, None

        best = top[0]
        target = None
        for a in self.asteroids:
            if a['x'] == best['x'] and a['y'] == best['y']:
                target = a
                break
        if target is None:
            return -0.1, False, None

        distance = self._calculate_distance(ship['x'], ship['y'], target['x'], target['y'])

        # Prevent jumping to same location (distance 0) -- this is a no-op
        if distance == 0:
            return -0.1, False, {'error': 'best asteroid is at current location'}

        energy_cost = int(distance * self.config['energy_costs']['jump'])

        if ship['energy'] < energy_cost:
            return -0.1, False, None

        # Jump to asteroid
        old_x, old_y = ship['x'], ship['y']
        ship['x'] = target['x']
        ship['y'] = target['y']
        ship['energy'] -= energy_cost

        payload = {'from': (old_x, old_y), 'to': (ship['x'], ship['y']), 'distance': distance, 'energy_cost': energy_cost}
        return -0.01, True, payload

    def _action_jump_to_trading_post(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Jump to nearest trading post"""
        if ship['recharging']:
            return -0.1, False, None

        # Find nearest trading post
        nearest_post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
        if nearest_post is None:
            return -0.1, False, None

        distance = self._calculate_distance(ship['x'], ship['y'], nearest_post['x'], nearest_post['y'])

        # Prevent jumping to same location (distance 0) -- should SELL instead
        if distance == 0:
            return -0.1, False, {'error': 'already at trading post, use SELL instead'}

        energy_cost = int(distance * self.config['energy_costs']['jump'])

        if ship['energy'] < energy_cost:
            return -0.1, False, None

        # Jump to trading post
        old_x, old_y = ship['x'], ship['y']
        ship['x'] = nearest_post['x']
        ship['y'] = nearest_post['y']
        ship['energy'] -= energy_cost

        payload = {'from': (old_x, old_y), 'to': (ship['x'], ship['y']), 'distance': distance, 'energy_cost': energy_cost}
        return -0.01, True, payload

    def _action_sell(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Sell nutrinium at a trading post"""
        if ship['recharging']:
            return -0.1, False, None

        # Check if at trading post
        trading_post = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
        if trading_post is None:
            return -0.1, False, None

        if ship['nutrinium'] <= 0:
            return -0.1, False, None

        # Sell all nutrinium
        nutrinium_sold = ship['nutrinium']
        credits_earned = nutrinium_sold * self.config['market']['nutrinium_price']
        ship['nutrinium'] = 0
        ship['credits'] += credits_earned

        payload = {'nutrinium_sold': nutrinium_sold, 'credits_earned': credits_earned}
        return credits_earned * 0.5, True, payload  # Reward for selling (increased from 0.1 to 0.5)

    def _action_respawn(self, ship: dict) -> bool:
        """Respawn a destroyed ship"""
        if not ship.get('destroyed', False):
            return False

        # Calculate respawn cost (increases each time)
        base_ship_cost = self.config['market']['ship_cost']
        respawn_count = ship.get('respawn_count', 0)
        respawn_cost = base_ship_cost * (respawn_count + 1)

        # Deduct cost from credits (can go negative)
        ship['credits'] -= respawn_cost

        # Reset ship state
        ship['destroyed'] = False
        ship['health'] = self.config['max_health']
        ship['energy'] = 0  # Start with 0 energy after respawn
        ship['nutrinium'] = 0  # Lose all nutrinium
        ship['shields_up'] = False
        ship['recharging'] = False
        ship['state'] = 'READY'

        # Respawn at random location
        ship['x'] = random.randint(0, self.map_width - 1)
        ship['y'] = random.randint(0, self.map_height - 1)

        # Increment respawn counter
        ship['respawn_count'] = respawn_count + 1

        return True

    def _action_attack(self, ship: dict, is_player: bool) -> Tuple[float, bool, Optional[dict]]:
        """Attack an enemy ship in the same zone.

        Player behavior: select the weakest active opponent (lowest combat score) as target.
        Opponent behavior: select the weakest ship in the same zone (player or other opponents).

        When a target is destroyed, transfer its nutrinium to the attacker automatically.
        """
        if ship['recharging']:
            return -0.1, False, None

        if ship['energy'] < self.config['energy_costs']['attack']:
            return -0.1, False, None

        # Determine targets depending on who is attacking
        if is_player:
            # Select weakest active opponent in the same zone
            same_zone_targets = [
                t for t in self.opponent_ships
                if not t.get('destroyed', False)
                and t['x'] == ship['x'] and t['y'] == ship['y']
            ]
            if not same_zone_targets:
                return -0.1, False, None

            # compute raw score for sorting (reuse _calculate_enemy_combat_score with raw=True)
            scored = [(self._calculate_enemy_combat_score(t, raw=True), t) for t in same_zone_targets]
            scored.sort(key=lambda x: x[0])  # ascending -> weakest first
            target = scored[0][1]
        else:
            # Opponent attacking: target any ship in the same zone (player or other opponents)
            # Build list of all potential targets in the same zone
            same_zone_targets = []

            # Consider the player as a target
            if (not self.player_ship.get('destroyed', False)
                    and self.player_ship['x'] == ship['x']
                    and self.player_ship['y'] == ship['y']):
                same_zone_targets.append(self.player_ship)

            # Consider other opponents as targets
            for other in self.opponent_ships:
                if other is ship:
                    continue  # Don't attack self
                if other.get('destroyed', False):
                    continue
                if other['x'] == ship['x'] and other['y'] == ship['y']:
                    same_zone_targets.append(other)

            if not same_zone_targets:
                return -0.1, False, None

            # Select the weakest target (lowest combat score)
            scored = [(self._calculate_enemy_combat_score(t, raw=True), t) for t in same_zone_targets]
            scored.sort(key=lambda x: x[0])  # ascending -> weakest first
            target = scored[0][1]

        if target is None:
            return -0.1, False, None

        # Mark both attacker and target as in combat, tracking opponent positions
        ship['in_combat'] = True
        ship.setdefault('combat_opponent_positions', set()).add((target['x'], target['y']))
        target['in_combat'] = True
        target.setdefault('combat_opponent_positions', set()).add((ship['x'], ship['y']))

        # Capture pre-combat state for detailed reporting
        attacker_abilities = ship.get('abilities', {})
        target_abilities = target.get('abilities', {})
        target_health_before = target['health']
        target_shields_up = target.get('shields_up', False)
        target_energy_before = target.get('energy', 0)
        attacker_energy_before = ship['energy']

        # Consume energy for attack
        attack_energy = self.config['energy_costs']['attack']
        ship['energy'] -= attack_energy

        # Calculate damage (simplified: base damage with some randomness)
        # Base damage is proportional to attack energy
        base_damage = attack_energy * 2  # 2 damage per energy unit
        damage_roll = base_damage * random.uniform(0.8, 1.2)
        damage = max(1, int(damage_roll))
        damage_before_shields = damage

        # Apply shield reduction if target has shields up
        shield_absorbed = 0
        if target_shields_up and target['energy'] > 0:
            shield_absorbed = damage - int(damage * 0.75)
            damage = int(damage * 0.75)  # 25% damage reduction with shields
            # Shields consume energy
            target['energy'] = max(0, target['energy'] - 1)

        # Apply damage to target
        target['health'] = max(0, target['health'] - damage)

        # Build detailed combat payload
        combat_details = {
            'target': target.get('name', 'Unknown'),
            'damage': damage,
            # Attacker stats
            'atk_energy': f"{attacker_energy_before}->{ship['energy']}",
            'atk_power': attacker_abilities.get('attack_power', 0),
            'atk_accuracy': attacker_abilities.get('attack_accuracy', 0),
            # Defender stats
            'def_health': f"{target_health_before}->{target['health']}",
            'def_shields': target_shields_up,
            'def_shield_str': target_abilities.get('shield_strength', 0),
            'def_evade': target_abilities.get('evade', 0),
            'def_energy': f"{target_energy_before}->{target.get('energy', 0)}",
            # Combat calculations
            'base_dmg': base_damage,
            'dmg_roll': round(damage_roll, 1),
            'shield_absorbed': shield_absorbed,
        }

        # Check if target is destroyed
        if target['health'] <= 0:
            target['destroyed'] = True
            target['state'] = 'DESTROYED'

            # Transfer nutrinium from destroyed ship to attacker
            nutrinium_stolen = target['nutrinium']
            ship['nutrinium'] += nutrinium_stolen
            target['nutrinium'] = 0

            # Reward for destroying enemy (modest - economic actions should be primary)
            reward = 0.5 + (nutrinium_stolen * 0.2)  # Reduced base from 2.0 to 0.5

            combat_details['destroyed'] = True
            combat_details['nutrinium_stolen'] = nutrinium_stolen

            return reward, True, combat_details
        else:
            # Successful hit but target survived
            # Reduced from 0.1 to 0.02 - attack micro-rewards were incentivizing
            # futile combat loops at low energy/HP (see episode_0003 analysis).
            # Combat should be a means to an end (steal nutrinium on kill), not a goal.
            reward = 0.02

            combat_details['destroyed'] = False
            combat_details['target_health'] = target['health']

            return reward, True, combat_details

    def _update_combat_states(self):
        """
        Update each ship's 'state' based on game state.

        COMBAT state should only be set when a ship is actively engaged in combat
        AND there is an enemy in the same zone. Ships attacked from a different zone
        should not remain in COMBAT state once the turn ends.

        This is called each step after all actions are executed.
        """
        # Build list of all ships (player + opponents)
        ships = [self.player_ship] + list(self.opponent_ships)

        for s in ships:
            if s is None:
                continue

            # Do not touch destroyed ships
            if s.get('destroyed', False):
                s['state'] = 'DESTROYED'
                continue

            # Recharging should remain RECHARGING
            if s.get('recharging', False):
                s['state'] = 'RECHARGING'
                continue

            # Combat state requires:
            # 1. The in_combat flag was set (an attack/shield action occurred)
            # 2. The combat was with an opponent at this ship's zone
            # 3. An enemy is STILL present in this zone (they may have moved away)
            combat_positions = s.get('combat_opponent_positions', set())
            was_same_zone_combat = s.get('in_combat', False) and (s['x'], s['y']) in combat_positions

            # Verify an enemy is still actually in the same zone
            if was_same_zone_combat:
                if s is self.player_ship:
                    enemies = self.opponent_ships
                else:
                    enemies = [self.player_ship] + [o for o in self.opponent_ships if o is not s]
                enemy_still_here = any(
                    e['x'] == s['x'] and e['y'] == s['y'] and not e.get('destroyed', False)
                    for e in enemies
                )
            else:
                enemy_still_here = False

            if was_same_zone_combat and enemy_still_here:
                s['state'] = 'COMBAT'
            else:
                # No same-zone combat this turn -> READY
                # Reset from any non-READY state (COMBAT, RECHARGING, etc.)
                if s.get('state', 'READY').upper() != 'READY':
                    s['state'] = 'READY'
                # Shields automatically go down when not in active same-zone combat
                if s.get('shields_up', False):
                    s['shields_up'] = False

            # Always clear combat flags for next turn
            s['in_combat'] = False
            s['combat_opponent_positions'] = set()

    def _update_passive_effects(self):
        """Update passive effects like recharging"""
        # Recharge player ship
        if self.player_ship['recharging'] and not self.player_ship['destroyed']:
            self.player_ship['energy'] = min(
                self.config['max_energy'],
                self.player_ship['energy'] + self.config['energy_per_recharge']
            )

        # Recharge opponents
        for ship in self.opponent_ships:
            if ship['recharging'] and not ship['destroyed']:
                ship['energy'] = min(
                    self.config['max_energy'],
                    ship['energy'] + self.config['energy_per_recharge']
                )

    def _get_opponent_action(self, ship: dict) -> int:
        """Dispatch opponent action based on assigned AI type"""
        ai_type = ship.get('ai_type', OpponentAIType.HEURISTIC)

        if ai_type == OpponentAIType.MODEL:
            return self._ai_model(ship)
        elif ai_type == OpponentAIType.PROSPECTOR:
            return self._ai_prospector(ship)
        elif ai_type == OpponentAIType.PIRATE:
            return self._ai_pirate(ship)
        else:  # HEURISTIC
            return self._ai_heuristic(ship)

    def _ai_model(self, ship: dict) -> int:
        """Model-based AI: Uses a trained RL model for decision-making.

        Args:
            ship: Enemy ship dictionary

        Returns:
            Action selected by the model (validated with action masking)
        """
        model_path = ship.get('model_path')

        if not model_path:
            logger.warning(f"MODEL AI type but no model_path specified for ship {ship.get('name')}. Falling back to HEURISTIC.")
            return self._ai_heuristic(ship)

        # Load model (uses cache if already loaded)
        model = self._load_enemy_model(model_path)

        if model is None:
            logger.warning(f"Failed to load model {model_path} for ship {ship.get('name')}. Falling back to HEURISTIC.")
            return self._ai_heuristic(ship)

        try:
            # Get observation from the enemy's perspective
            # We need to create an observation as if this enemy was the player
            # For simplicity, we'll use the same observation space but from enemy's viewpoint
            obs = self._get_enemy_observation(ship)

            # Predict action using the model
            # Detect what observation format the loaded model expects:
            #   - Old models trained with Box obs space -> pass flat ndarray
            #   - New models trained with Dict obs space -> pass full dict
            from gymnasium import spaces as _spaces
            model_obs_space = getattr(model, 'observation_space', None)

            if isinstance(model_obs_space, _spaces.Dict):
                # Model expects Dict observation -- pass full dict, truncating obs if needed
                obs_for_model = obs
                if (isinstance(obs, dict) and 'observation' in obs and
                        'observation' in model_obs_space.spaces):
                    model_obs_size = model_obs_space['observation'].shape[0]
                    if obs['observation'].shape[0] != model_obs_size:
                        obs_for_model = dict(obs)
                        obs_for_model['observation'] = obs['observation'][:model_obs_size]
            else:
                # Model expects flat Box observation -- extract the array
                if isinstance(obs, dict):
                    obs_for_model = obs['observation']
                else:
                    obs_for_model = obs
                # Truncate if model expects fewer features
                if (model_obs_space is not None and
                        isinstance(model_obs_space, _spaces.Box) and
                        obs_for_model.shape[0] != model_obs_space.shape[0]):
                    obs_for_model = obs_for_model[:model_obs_space.shape[0]]

            action, _ = model.predict(obs_for_model, deterministic=True)

            # Convert to int (handle numpy arrays)
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.size == 1 else int(action[0])
            else:
                action = int(action)

            # Validate and enforce action masking for MODEL enemies
            # Without this, models waste turns on invalid actions (ATTACK with no enemy,
            # SELL with no trading post, MINE with no asteroid, etc.)
            action = self._enforce_enemy_action_mask(ship, action)

            return action

        except Exception as e:
            logger.warning(f"Error using model for ship {ship.get('name')}: {e}. Falling back to HEURISTIC.")
            return self._ai_heuristic(ship)

    def _enforce_enemy_action_mask(self, ship: dict, action: int) -> int:
        """Enforce action masking for an enemy ship, replacing invalid actions with valid ones.

        This gives MODEL enemies the same action enforcement the player gets,
        preventing them from wasting turns on invalid actions.
        """
        is_valid, reason = self._is_action_valid_for_state(action, ship, is_player=False)
        if is_valid:
            return action

        # Invalid action - apply fallback logic similar to player enforcement
        if ship.get('recharging', False):
            if ship['energy'] >= self.config['max_energy']:
                return int(ActionType.RECHARGE_END)
            elif action not in (int(ActionType.WAIT), int(ActionType.RECHARGE_END)):
                return int(ActionType.RECHARGE_END)
            else:
                return int(ActionType.WAIT)
        elif ship.get('destroyed', False):
            return int(ActionType.RESPAWN)
        else:
            # Pick the best valid action from the action mask
            mask = self._get_action_mask(ship, is_player=False)
            if ship['energy'] <= self.config['energy_costs'].get('move', 5):
                preferred = [
                    ActionType.RECHARGE, ActionType.MINE, ActionType.SELL,
                    ActionType.WAIT, ActionType.JUMP_TO_ASTEROID,
                    ActionType.JUMP_TO_TRADING_POST,
                    ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                    ActionType.MOVE_EAST, ActionType.MOVE_WEST,
                    ActionType.ATTACK, ActionType.RAISE_SHIELDS,
                ]
            else:
                preferred = [
                    ActionType.MINE, ActionType.SELL,
                    ActionType.JUMP_TO_ASTEROID, ActionType.JUMP_TO_TRADING_POST,
                    ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                    ActionType.MOVE_EAST, ActionType.MOVE_WEST,
                    ActionType.RECHARGE, ActionType.ATTACK,
                    ActionType.RAISE_SHIELDS, ActionType.WAIT,
                ]
            for fb in preferred:
                if mask[int(fb)] == 1:
                    return int(fb)
            return int(ActionType.WAIT)

    def _get_enemy_observation(self, enemy_ship: dict) -> Dict[str, np.ndarray]:
        """
        Get observation from an enemy ship's perspective.

        This creates an observation as if the enemy was the player,
        allowing us to use player-trained models for enemy AI.

        Args:
            enemy_ship: The enemy ship dictionary

        Returns:
            Dict observation compatible with the environment's observation space
        """
        # Temporarily swap player and enemy to get enemy's perspective
        original_player = self.player_ship
        original_opponents = self.opponent_ships

        try:
            # Create a temporary opponent list (current player + other enemies, excluding this enemy)
            temp_opponents = [original_player] + [s for s in original_opponents if s != enemy_ship]

            # Temporarily set this enemy as the "player"
            self.player_ship = enemy_ship
            self.opponent_ships = temp_opponents

            # Get observation WITH proper action mask (skip_mask=False)
            # This is critical: the model was trained with action masking and needs it
            # to make proper predictions. Using skip_mask=True gives a dummy all-1s mask
            # which causes the model to make completely different (suboptimal) predictions.
            obs = self._get_observation(skip_mask=False)

            return obs

        finally:
            # Restore original state
            self.player_ship = original_player
            self.opponent_ships = original_opponents

    def _ai_prospector(self, ship: dict) -> int:
        """Prospector AI: Optimised mining-selling loop with efficient travel.

        Key principles:
        - Maximise the number of mine->sell cycles completed per episode
        - Minimise travel time by choosing asteroids near trading posts
        - Jump aggressively instead of walking (saves many turns)
        - Sell ANY cargo at a trading post -- even small amounts are free credits
        - Keep energy lean: short recharge cycles, don't over-charge
        - Never fight -- pure economy
        """
        # === 1. ENERGY MANAGEMENT ===
        if ship.get('recharging', False):
            # Short recharge: get back to work quickly
            if ship['energy'] >= 60:
                return ActionType.RECHARGE_END
            return ActionType.WAIT

        # Recharge only when truly low -- every recharge turn is a lost mining turn
        if ship['energy'] < 15 and not ship.get('just_recharged', False):
            return ActionType.RECHARGE

        # === 2. SELL at trading post -- always, any amount ===
        trading_post = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
        if trading_post and ship['nutrinium'] > 0:
            return ActionType.SELL

        # === 3. HEAD TO TRADING POST when carrying cargo ===
        if ship['nutrinium'] >= 12:
            nearest_post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
            if nearest_post:
                dist = self._calculate_distance(ship['x'], ship['y'], nearest_post['x'], nearest_post['y'])
                jump_cost = int(dist * self.config['energy_costs']['jump'])

                # Jump to trading post aggressively (even short distances)
                if dist > 1 and ship['energy'] >= jump_cost + 5:
                    return ActionType.JUMP_TO_TRADING_POST

                # Walk to trading post
                dx = nearest_post['x'] - ship['x']
                dy = nearest_post['y'] - ship['y']
                if abs(dx) > abs(dy):
                    return ActionType.MOVE_EAST if dx > 0 else ActionType.MOVE_WEST
                else:
                    return ActionType.MOVE_SOUTH if dy > 0 else ActionType.MOVE_NORTH

        # === 4. MINE current asteroid ===
        asteroid = self._get_entity_at_location(ship['x'], ship['y'], self.asteroids)
        if (asteroid and asteroid['nutrinium'] > 0
                and ship['energy'] >= self.config['energy_costs']['mine']):
            return ActionType.MINE

        # === 5. FIND BEST ASTEROID (optimised for round-trip efficiency) ===
        best_asteroid = None
        best_score = -1
        for ast in self.asteroids:
            if ast['nutrinium'] <= 0:
                continue

            dist_to_ast = self._calculate_distance(ship['x'], ship['y'], ast['x'], ast['y'])

            # Factor in distance from asteroid to nearest trading post
            nearest_post = self._get_nearest_entity(ast['x'], ast['y'], self.trading_posts)
            dist_to_post = 10.0
            if nearest_post:
                dist_to_post = self._calculate_distance(
                    ast['x'], ast['y'], nearest_post['x'], nearest_post['y'])

            # Round-trip cost: getting there + getting to trading post after
            round_trip = dist_to_ast + dist_to_post * 0.6

            # Score: favour rich asteroids (nutrinium^1.3), penalise by round-trip distance
            score = (ast['nutrinium'] ** 1.3) / (round_trip + 1)
            if score > best_score:
                best_score = score
                best_asteroid = ast

        if best_asteroid:
            dist = self._calculate_distance(ship['x'], ship['y'],
                                            best_asteroid['x'], best_asteroid['y'])
            jump_cost = int(dist * self.config['energy_costs']['jump'])

            # Jump aggressively -- walking wastes turns
            if dist > 2 and ship['energy'] >= jump_cost + 10:
                return ActionType.JUMP_TO_ASTEROID

            # Walk towards asteroid
            dx = best_asteroid['x'] - ship['x']
            dy = best_asteroid['y'] - ship['y']
            if abs(dx) > abs(dy):
                return ActionType.MOVE_EAST if dx > 0 else ActionType.MOVE_WEST
            else:
                return ActionType.MOVE_SOUTH if dy > 0 else ActionType.MOVE_NORTH

        # === 6. SELL remaining cargo if nothing to mine ===
        if ship['nutrinium'] > 0:
            nearest_post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
            if nearest_post:
                dist = self._calculate_distance(ship['x'], ship['y'],
                                                nearest_post['x'], nearest_post['y'])
                jump_cost = int(dist * self.config['energy_costs']['jump'])
                if dist > 1 and ship['energy'] >= jump_cost + 5:
                    return ActionType.JUMP_TO_TRADING_POST
                dx = nearest_post['x'] - ship['x']
                dy = nearest_post['y'] - ship['y']
                if abs(dx) > abs(dy):
                    return ActionType.MOVE_EAST if dx > 0 else ActionType.MOVE_WEST
                else:
                    return ActionType.MOVE_SOUTH if dy > 0 else ActionType.MOVE_NORTH

        return ActionType.WAIT

    def _ai_pirate(self, ship: dict) -> int:
        """Pirate AI: Economy-first raider -- mines efficiently, only strikes opportunistically.

        The game's combat deals ~2 damage per attack (cost: 1 energy). Killing a 100HP
        target takes ~50 attacks = 50 turns of mutual damage. Pure combat is suicide.

        Winning strategy: Be a top-tier miner/seller who ALSO finishes off weak targets
        to steal their nutrinium cargo.

        Priority order:
        1. Energy management (never run dry)
        2. Sell nutrinium (cash is safe, nutrinium can be stolen)
        3. Finish off near-dead targets in same zone (steal their cargo)
        4. Flee if health is dangerously low
        5. Mine asteroids efficiently (jump to rich ones near trading posts)
        6. Head to trading post when cargo is ready
        """
        # === 1. ENERGY MANAGEMENT ===
        if ship.get('recharging', False):
            if ship['energy'] >= 70:
                return ActionType.RECHARGE_END
            return ActionType.WAIT

        if ship['energy'] < 15 and not ship.get('just_recharged', False):
            return ActionType.RECHARGE

        # === 2. ALWAYS SELL at trading post ===
        trading_post = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
        if trading_post and ship['nutrinium'] > 0:
            return ActionType.SELL

        # === 3. SURGICAL STRIKES: Only attack high-value targets in same zone ===
        same_zone_targets = []
        if (not self.player_ship.get('destroyed', False)
                and self.player_ship['x'] == ship['x']
                and self.player_ship['y'] == ship['y']):
            same_zone_targets.append(self.player_ship)
        for other in self.opponent_ships:
            if other is ship or other.get('destroyed', False):
                continue
            if other['x'] == ship['x'] and other['y'] == ship['y']:
                same_zone_targets.append(other)

        if same_zone_targets and ship['energy'] >= self.config['energy_costs']['attack']:
            # Find the most profitable target in zone
            best_value = 0
            best_target_health = 100
            for t in same_zone_targets:
                t_nutr = t.get('nutrinium', 0)
                t_health = t.get('health', 100)
                # Attack priority (more aggressive than before):
                # 1) Target at low health (finishable) - always attack
                # 2) Target with valuable cargo even if healthy
                # 3) Weaken any target in zone if we're strong
                if t_health <= 10:
                    # Can kill in 5 attacks or less -- finish them off
                    value = 60 + t_nutr * 3
                elif t_health <= 25:
                    # Can kill in ~12 attacks - worth it if they have cargo
                    value = 40 + t_nutr * 2
                elif t_nutr >= 15:
                    # They have valuable cargo - attack even if healthy
                    # (weaken them so we can finish later or steal if they run)
                    value = 25 + t_nutr * 1.5
                elif t_nutr >= 8 and t_health <= 60:
                    # Moderate cargo, weakened - opportunistic strike
                    value = 15 + t_nutr
                elif t_health <= 40:
                    # Weaken any target below 40% health
                    value = 10 + t_nutr * 0.5
                else:
                    value = 0  # Full-health targets with no cargo not worth it

                if value > best_value:
                    best_value = value
                    best_target_health = t_health

            # Attack if target is valuable AND we're healthy enough
            # More willing to fight if target is already weakened
            health_threshold = 30 if best_target_health > 50 else 20  # Lower bar for attacking weak targets
            if best_value > 10 and ship['health'] > health_threshold:
                return ActionType.ATTACK

        # === 4. FLEE if health is low and enemies are nearby ===
        if ship['health'] < 40 and same_zone_targets:
            # Move away from enemies -- pick a direction away from the nearest threat
            threat = same_zone_targets[0]
            dx = ship['x'] - threat['x']
            dy = ship['y'] - threat['y']
            # Move in the opposite direction; if at same spot, pick a random cardinal
            if dx == 0 and dy == 0:
                # Move toward nearest asteroid or trading post as escape
                escape = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
                if escape is None:
                    escape = self._get_nearest_entity(ship['x'], ship['y'], self.asteroids)
                if escape:
                    dx = escape['x'] - ship['x']
                    dy = escape['y'] - ship['y']
                else:
                    dx, dy = 1, 0  # default east
            if abs(dx) >= abs(dy):
                return ActionType.MOVE_EAST if dx > 0 else ActionType.MOVE_WEST
            else:
                return ActionType.MOVE_SOUTH if dy > 0 else ActionType.MOVE_NORTH

        # === 5. PRIMARY ECONOMY: Mine and sell efficiently ===
        # If on asteroid with nutrinium, mine it
        asteroid = self._get_entity_at_location(ship['x'], ship['y'], self.asteroids)
        if asteroid and asteroid['nutrinium'] > 0 and ship['energy'] >= self.config['energy_costs']['mine']:
            return ActionType.MINE

        # === 6. HEAD TO TRADING POST when carrying cargo ===
        if ship['nutrinium'] >= 12:
            nearest_post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
            if nearest_post:
                dist = self._calculate_distance(ship['x'], ship['y'], nearest_post['x'], nearest_post['y'])
                jump_cost = int(dist * self.config['energy_costs']['jump'])

                if dist > 1 and ship['energy'] >= jump_cost + 5:
                    return ActionType.JUMP_TO_TRADING_POST

                dx = nearest_post['x'] - ship['x']
                dy = nearest_post['y'] - ship['y']
                if abs(dx) > abs(dy):
                    return ActionType.MOVE_EAST if dx > 0 else ActionType.MOVE_WEST
                else:
                    return ActionType.MOVE_SOUTH if dy > 0 else ActionType.MOVE_NORTH

        # === 7. FIND BEST ASTEROID (prefer rich ones near trading posts) ===
        best_asteroid = None
        best_ast_score = -1
        for ast in self.asteroids:
            if ast['nutrinium'] <= 0:
                continue
            dist_to_ast = self._calculate_distance(ship['x'], ship['y'], ast['x'], ast['y'])

            # Factor in proximity to nearest trading post (round-trip efficiency)
            nearest_post = self._get_nearest_entity(ast['x'], ast['y'], self.trading_posts)
            dist_ast_to_post = 10.0
            if nearest_post:
                dist_ast_to_post = self._calculate_distance(ast['x'], ast['y'], nearest_post['x'], nearest_post['y'])

            # Score: nutrinium value vs total travel cost
            total_travel = dist_to_ast + dist_ast_to_post * 0.5
            score = (ast['nutrinium'] ** 1.3) / (total_travel + 1)
            if score > best_ast_score:
                best_ast_score = score
                best_asteroid = ast

        if best_asteroid:
            dist = self._calculate_distance(ship['x'], ship['y'], best_asteroid['x'], best_asteroid['y'])
            jump_cost = int(dist * self.config['energy_costs']['jump'])

            if dist > 2 and ship['energy'] >= jump_cost + 15:
                return ActionType.JUMP_TO_ASTEROID

            dx = best_asteroid['x'] - ship['x']
            dy = best_asteroid['y'] - ship['y']
            if abs(dx) > abs(dy):
                return ActionType.MOVE_EAST if dx > 0 else ActionType.MOVE_WEST
            else:
                return ActionType.MOVE_SOUTH if dy > 0 else ActionType.MOVE_NORTH

        return ActionType.WAIT

    def _ai_heuristic(self, ship: dict) -> int:
        """Heuristic AI: Balanced approach with smart decision-making.

        Strategy:
        - Balance between mining and combat
        - Attack weak enemies when opportune
        - Smart resource management
        - Adaptive behavior based on game state
        """
        # Energy management - balanced threshold
        if ship['energy'] < 20 and not ship['recharging']:
            return ActionType.RECHARGE

        if ship['recharging'] and ship['energy'] > 80:
            return ActionType.RECHARGE_END

        # Sell at moderate threshold
        trading_post = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
        if trading_post and ship['nutrinium'] > 25:
            return ActionType.SELL

        # Opportunistic combat - attack if enemy is nearby, weak, and has nutrinium
        if ship['energy'] > 30:  # Only consider combat if we have decent energy
            for enemy in [self.player_ship] + [s for s in self.opponent_ships if s != ship]:
                if enemy.get('destroyed', False):
                    continue

                dist = self._calculate_distance(ship['x'], ship['y'], enemy['x'], enemy['y'])

                # Attack if: same zone, enemy is weak, and enemy has nutrinium
                if (dist == 0 and
                    enemy.get('health', 100) < 50 and
                    enemy.get('nutrinium', 0) > 15):
                    return ActionType.ATTACK

        # If on asteroid with good nutrinium, mine it
        asteroid = self._get_entity_at_location(ship['x'], ship['y'], self.asteroids)
        if asteroid and asteroid['nutrinium'] > 0 and ship['energy'] >= self.config['energy_costs']['mine']:
            # Mine if it's a decent asteroid
            if asteroid['nutrinium'] > 5:
                return ActionType.MINE

        # Smart asteroid selection - consider both distance and nutrinium amount
        best_asteroid = None
        best_score = -1
        for ast in self.asteroids:
            if ast['nutrinium'] <= 0:
                continue
            dist = self._calculate_distance(ship['x'], ship['y'], ast['x'], ast['y'])
            # Balanced score: nutrinium value vs distance
            score = (ast['nutrinium'] ** 1.5) / (dist + 2)  # Favor richer asteroids more
            if score > best_score:
                best_score = score
                best_asteroid = ast

        if best_asteroid:
            distance = self._calculate_distance(ship['x'], ship['y'], best_asteroid['x'], best_asteroid['y'])
            jump_cost = int(distance * self.config['energy_costs']['jump'])

            # Smart jump decision: jump if distance is significant and we have spare energy
            if ship['energy'] >= jump_cost + 15 and distance > 5 and best_asteroid['nutrinium'] > 15:
                return ActionType.JUMP_TO_ASTEROID
            else:
                # Move towards asteroid
                dx = best_asteroid['x'] - ship['x']
                dy = best_asteroid['y'] - ship['y']
                if abs(dx) > abs(dy):
                    return ActionType.MOVE_EAST if dx > 0 else ActionType.MOVE_WEST
                else:
                    return ActionType.MOVE_SOUTH if dy > 0 else ActionType.MOVE_NORTH

        # If we have nutrinium and energy, head towards trading post
        if ship['nutrinium'] > 15 and ship['energy'] > 25:
            nearest_post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
            if nearest_post:
                dx = nearest_post['x'] - ship['x']
                dy = nearest_post['y'] - ship['y']
                if abs(dx) > abs(dy):
                    return ActionType.MOVE_EAST if dx > 0 else ActionType.MOVE_WEST
                else:
                    return ActionType.MOVE_SOUTH if dy > 0 else ActionType.MOVE_NORTH

        return ActionType.WAIT

    def _get_observation(self, skip_mask: bool = False) -> Dict[str, np.ndarray]:
        """Get the current observation with enhanced ship state and entity info.

        Args:
            skip_mask: If True, return a dummy action mask (all ones) to save computation.
                       Used for enemy observations where the mask is not needed.
        """
        obs = []
        ship = self.player_ship
        abilities = ship.get('abilities', {})
        max_abilities = self.config.get('abilities', {})

        # === ENHANCED SHIP STATE (24 values) ===
        # Basic stats (6 values)
        obs.extend([
            ship['x'] / max(1, self.map_width),
            ship['y'] / max(1, self.map_height),
            ship['energy'] / max(1, self.config['max_energy']),
            ship['health'] / max(1, self.config['max_health']),
            min(ship['nutrinium'], self.config['max_nutrinium_cargo']) / max(1, self.config['max_nutrinium_cargo']),
            min(ship['credits'], self.config['max_credits']) / max(1, self.config['max_credits']),
        ])

        # State flags (3 values)
        obs.extend([
            1.0 if ship.get('recharging', False) else 0.0,
            1.0 if ship.get('shields_up', False) else 0.0,
            1.0 if ship.get('state', 'READY') == 'READY' else 0.0,
        ])

        # Skill points (2 values)
        obs.extend([
            ship.get('skill_points_total', 5) / max(1, self.config.get('max_skill_points', 20)),
            ship.get('skill_points_spent', 0) / max(1, self.config.get('max_skill_points', 20)),
        ])

        # Abilities (12 values)
        obs.extend([
            abilities.get('energy_max', 5) / max(1, max_abilities.get('energy_max', 10)),
            abilities.get('recharge_energy', 0) / max(1, max_abilities.get('recharge_energy', 10)),
            abilities.get('mine_accuracy', 0) / max(1, max_abilities.get('mine_accuracy', 10)),
            abilities.get('mine_yield_multiplier', 1) / max(1, max_abilities.get('mine_yield_multiplier', 5)),
            abilities.get('mine_cost', 2) / max(1, max_abilities.get('mine_cost', 10)),
            abilities.get('combat_salvage_multiplier', 0) / max(1, max_abilities.get('combat_salvage_multiplier', 5)),
            abilities.get('sensor_range', 1) / max(1, self.config['sensor_range']),
            abilities.get('attack_accuracy', 0) / max(1, max_abilities.get('attack_accuracy', 10)),
            abilities.get('attack_power', 0) / max(1, max_abilities.get('attack_power', 10)),
            abilities.get('evade', 0) / max(1, max_abilities.get('evade', 10)),
            abilities.get('shield_strength', 0) / max(1, max_abilities.get('shield_strength', 10)),
            abilities.get('jump_distance', 0) / max(1, max_abilities.get('jump_distance', 10)),
        ])

        # Action counter (1 value) - normalized by max_steps (typical ~300)
        obs.append(self.action_counter / max(1, self.max_steps))

        # === STRATEGIC CONTEXT (8 values) ===
        # These high-signal features directly encode actionable state
        map_diag = max(1.0, math.sqrt(self.map_width**2 + self.map_height**2))

        # 1. At asteroid with nutrinium?
        ast_here = self._get_entity_at_location(ship['x'], ship['y'], self.asteroids)
        obs.append(1.0 if (ast_here and ast_here.get('nutrinium', 0) > 0) else 0.0)

        # 2. At trading post?
        tp_here = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
        obs.append(1.0 if tp_here else 0.0)

        # 3. Cargo fullness (nutrinium as fraction of a "sell-worthy" amount ~25)
        obs.append(min(1.0, ship.get('nutrinium', 0) / 25.0))

        # 4. Enemy in same zone?
        enemy_here = any(
            e['x'] == ship['x'] and e['y'] == ship['y'] and not e.get('destroyed', False)
            for e in self.opponent_ships
        )
        obs.append(1.0 if enemy_here else 0.0)

        # 5-6. Direction to best asteroid (dx, dy normalized to [-1, 1])
        top_ast = self._get_top_asteroids(ship['x'], ship['y'], count=1)
        if top_ast:
            dx_ast = (top_ast[0]['x'] - ship['x']) / max(1, self.map_width)
            dy_ast = (top_ast[0]['y'] - ship['y']) / max(1, self.map_height)
        else:
            dx_ast, dy_ast = 0.0, 0.0
        obs.extend([dx_ast, dy_ast])

        # 7-8. Direction to nearest trading post (dx, dy normalized to [-1, 1])
        nearest_tp = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
        if nearest_tp:
            dx_tp = (nearest_tp['x'] - ship['x']) / max(1, self.map_width)
            dy_tp = (nearest_tp['y'] - ship['y']) / max(1, self.map_height)
        else:
            dx_tp, dy_tp = 0.0, 0.0
        obs.extend([dx_tp, dy_tp])

        # === LOCAL SENSOR GRID (with clamped/shifted window to maximize valid cells) ===
        sensor_range = self.config['sensor_range']
        side = 2 * sensor_range + 1  # Grid dimension (e.g., 11 for sensor_range=5)

        # Calculate top-left corner of a centered window
        x_min = ship['x'] - sensor_range
        y_min = ship['y'] - sensor_range

        # Clamp window to stay within map bounds (shifts window when near edges)
        # This maximizes the number of valid cells in the observation
        x_min = max(0, min(x_min, self.map_width - side)) if self.map_width >= side else 0
        y_min = max(0, min(y_min, self.map_height - side)) if self.map_height >= side else 0

        # Fill the sensor grid in row-major order (same as before for consistency)
        for row in range(side):
            for col in range(side):
                x = x_min + col
                y = y_min + row

                # Check if coordinate is valid (should almost always be true with clamping)
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    # Default: empty cell
                    entity_type = 0.0

                    # Player's own cell remains 0.0 (empty)
                    if x == ship['x'] and y == ship['y']:
                        entity_type = 0.0
                    # Check for entities (priority: enemy > trading_post > asteroid)
                    elif self._get_entity_at_location(x, y, self.opponent_ships):
                        entity_type = 1.0
                    elif self._get_entity_at_location(x, y, self.trading_posts):
                        entity_type = 0.66
                    elif self._get_entity_at_location(x, y, self.asteroids):
                        entity_type = 0.33

                    obs.append(entity_type)
                else:
                    # Out of bounds (should be rare with clamping, only when map < sensor grid)
                    obs.append(-1.0)

        # === TOP 5 ASTEROIDS (30 values: 5 asteroids * 6 features) ===
        top_asteroids = self._get_top_asteroids(ship['x'], ship['y'], count=self.config['top_asteroids_count'])
        max_dist = math.sqrt(self.map_width**2 + self.map_height**2)
        max_mass = float(self.config.get('asteroid_mass_max', 80))

        for asteroid in top_asteroids:
            obs.extend([
                asteroid['x'] / max(1, self.map_width),
                asteroid['y'] / max(1, self.map_height),
                asteroid['mass'] / max(1.0, max_mass),
                asteroid['nutrinium'] / max(1.0, max_mass),
                asteroid['distance'] / max(1.0, max_dist),
                asteroid['score'],  # Already normalized 0-1
            ])

        # Pad with zeros if fewer than 5 asteroids
        for _ in range(self.config['top_asteroids_count'] - len(top_asteroids)):
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # === NEAREST TRADING POST (3 values) ===
        nearest_post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
        if nearest_post:
            dist = self._calculate_distance(ship['x'], ship['y'], nearest_post['x'], nearest_post['y'])
            obs.extend([
                nearest_post['x'] / max(1, self.map_width),
                nearest_post['y'] / max(1, self.map_height),
                dist / max(1, max_dist),
            ])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # === TWO ENEMY TYPES (14 values: 2 enemies * 7 features) ===
        # Get strongest and weakest enemies at same coordinates as player
        strongest, weakest = self._get_extreme_enemies(ship['x'], ship['y'])

        for enemy in [strongest, weakest]:
            if enemy:
                combat_score = self._calculate_enemy_combat_score(enemy)
                obs.extend([
                    enemy['x'] / max(1, self.map_width),
                    enemy['y'] / max(1, self.map_height),
                    enemy['energy'] / max(1, self.config['max_energy']),
                    enemy['health'] / max(1, self.config['max_health']),
                    min(enemy['nutrinium'], 100) / 100.0,
                    min(enemy['credits'], 1000) / 1000.0,
                    combat_score,  # Already normalized 0-1
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Return Dict observation with action mask
        obs_array = np.array(obs, dtype=np.float32)
        if skip_mask:
            mask = np.ones(self.action_space.n, dtype=np.int8)
        else:
            mask = self._get_action_mask(ship)
        return {
            'observation': obs_array,
            'action_mask': mask
        }

    def _get_top_asteroids(self, x: int, y: int, count: int = 5) -> List[dict]:
        """
        Get top N asteroids ranked by a score combining mass, nutrinium concentration, and distance.

        Score formula: (nutrinium / mass) * nutrinium / (distance + 1)
        Higher score = better asteroid to target
        """
        if not self.asteroids:
            return []

        max_dist = math.sqrt(self.map_width**2 + self.map_height**2)
        scored_asteroids = []

        for asteroid in self.asteroids:
            if asteroid.get('nutrinium', 0) <= 0:
                continue

            dist = self._calculate_distance(x, y, asteroid['x'], asteroid['y'])
            mass = max(1, asteroid.get('mass', 1))
            nutrinium = asteroid.get('nutrinium', 0)

            # Calculate concentration (nutrinium / mass)
            concentration = nutrinium / mass

            # Score: concentration * nutrinium / (distance + 1)
            # This prioritizes: high concentration, high nutrinium, low distance
            raw_score = concentration * nutrinium / (dist + 1)

            # Normalize score to 0-1 range (approximate max score)
            max_score = 50.0  # Reasonable max for normalization
            normalized_score = min(1.0, raw_score / max_score)

            scored_asteroids.append({
                'x': asteroid['x'],
                'y': asteroid['y'],
                'mass': asteroid['mass'],
                'nutrinium': asteroid['nutrinium'],
                'distance': dist,
                'score': normalized_score,
            })

        # Sort by score descending and return top N
        scored_asteroids.sort(key=lambda a: a['score'], reverse=True)
        return scored_asteroids[:count]

    def _get_extreme_enemies(self, x: int, y: int) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Get the strongest and weakest active enemies.

        Strongest: highest combined health, energy, credits, and combat abilities
        Weakest: lowest combined values

        Returns: (strongest_enemy, weakest_enemy) - both at same coordinates as player for observation
        """
        active_enemies = [s for s in self.opponent_ships if not s.get('destroyed', False)]

        if not active_enemies:
            return None, None

        if len(active_enemies) == 1:
            return active_enemies[0], active_enemies[0]

        # Score each enemy
        scored_enemies = []
        for enemy in active_enemies:
            score = self._calculate_enemy_combat_score(enemy, raw=True)
            scored_enemies.append((score, enemy))

        # Sort by score
        scored_enemies.sort(key=lambda x: x[0], reverse=True)

        strongest = scored_enemies[0][1]
        weakest = scored_enemies[-1][1]

        return strongest, weakest

    def _calculate_enemy_combat_score(self, enemy: dict, raw: bool = False) -> float:
        """
        Calculate a combat score for an enemy ship.

        Factors: health, energy, credits, attack_power, attack_accuracy, shield_strength

        Args:
            enemy: Enemy ship dictionary
            raw: If True, return raw score; otherwise return normalized 0-1 score
        """
        health = enemy.get('health', 0)
        energy = enemy.get('energy', 0)
        credits = enemy.get('credits', 0)
        abilities = enemy.get('abilities', {})

        attack_power = abilities.get('attack_power', 0)
        attack_accuracy = abilities.get('attack_accuracy', 0)
        shield_strength = abilities.get('shield_strength', 0)
        evade = abilities.get('evade', 0)

        # Weighted score
        raw_score = (
            health * 1.0 +
            energy * 0.5 +
            credits * 0.1 +
            attack_power * 10.0 +
            attack_accuracy * 5.0 +
            shield_strength * 8.0 +
            evade * 3.0
        )

        if raw:
            return raw_score

        # Normalize (approximate max score)
        max_score = 100 + 50 + 100 + 100 + 50 + 80 + 30  # ~510
        return min(1.0, raw_score / max_score)

    def _get_info(self) -> dict:
        """Get additional information about the current state"""
        return {
            'step': self.current_step,
            'action_counter': self.action_counter,  # Track actions taken this episode
            'player_credits': self.player_ship['credits'],
            'player_nutrinium': self.player_ship['nutrinium'],
            'player_energy': self.player_ship['energy'],
            'player_health': self.player_ship['health'],
            'player_destroyed': self.player_ship['destroyed'],
            'asteroids_remaining': len([a for a in self.asteroids if a['nutrinium'] > 0]),
            'opponents_alive': len([s for s in self.opponent_ships if not s['destroyed']]),
            'invalid_action_count': getattr(self, 'invalid_action_count', 0),
            'state_invalid_action_count': getattr(self, 'state_invalid_action_count', 0),
        }

    def _get_entity_at_location(self, x: int, y: int, entities: List[dict]) -> Optional[dict]:
        """Get entity at a specific location using spatial cache for performance."""
        # Try spatial cache first (keyed by list identity)
        cache = getattr(self, '_entity_location_cache', None)
        if cache is not None:
            # Use named keys for the canonical lists
            if entities is self.asteroids:
                loc_map = cache.get('asteroids')
            elif entities is self.trading_posts:
                loc_map = cache.get('trading_posts')
            elif entities is self.opponent_ships:
                loc_map = cache.get('opponents')
            else:
                loc_map = None
            if loc_map is not None:
                return loc_map.get((x, y))
        # Fallback: linear scan
        for entity in entities:
            if entity['x'] == x and entity['y'] == y:
                if entity.get('destroyed', False):
                    continue
                return entity
        return None

    def _rebuild_location_cache(self):
        """Rebuild spatial lookup cache for all entity lists. Call once per step."""
        self._entity_location_cache = {}
        for name, entities in (('asteroids', self.asteroids), ('trading_posts', self.trading_posts), ('opponents', self.opponent_ships)):
            loc_map = {}
            for entity in entities:
                if entity.get('destroyed', False):
                    continue
                key = (entity['x'], entity['y'])
                if key not in loc_map:
                    loc_map[key] = entity
            self._entity_location_cache[name] = loc_map

    def _get_nearest_entity(self, x: int, y: int, entities: List[dict]) -> Optional[dict]:
        """Get nearest entity to a location"""
        if not entities:
            return None

        min_dist = float('inf')
        nearest = None

        for entity in entities:
            if 'destroyed' in entity and entity['destroyed']:
                continue
            if 'nutrinium' in entity and entity['nutrinium'] <= 0:
                continue

            dist = self._calculate_distance(x, y, entity['x'], entity['y'])
            if dist < min_dist:
                min_dist = dist
                nearest = entity

        return nearest

    def _calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            self._render_text()

    def _render_text(self):
        """Render the game state as text"""
        print(f"\n{'=' * 90}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"{'=' * 90}")

        # Create map with wider spacing
        game_map = [['  ' for _ in range(self.map_width)] for _ in range(self.map_height)]

        # Place entities with more information
        # Store asteroid info for displaying nutrinium
        asteroid_map = {}
        for asteroid in self.asteroids:
            if asteroid['nutrinium'] > 0:
                x, y = asteroid['x'], asteroid['y']
                # Use numbers to represent asteroid size/nutrinium
                if asteroid['nutrinium'] > 40:
                    game_map[y][x] = '* '  # Large asteroid
                elif asteroid['nutrinium'] > 20:
                    game_map[y][x] = '* '  # Medium asteroid
                else:
                    game_map[y][x] = 'o '  # Small asteroid
                asteroid_map[(x, y)] = asteroid

        for post in self.trading_posts:
            game_map[post['y']][post['x']] = 'T '

        # Store enemy info
        enemy_map = {}
        for i, ship in enumerate(self.opponent_ships):
            if not ship['destroyed']:
                x, y = ship['x'], ship['y']
                game_map[y][x] = f"{i + 1} "  # Number enemies
                enemy_map[i] = ship

        if not self.player_ship['destroyed']:
            game_map[self.player_ship['y']][self.player_ship['x']] = 'P '

        # Print map with wider layout
        # Build a per-cell listing of entities so we can show multiple entities comma-separated
        cell_entities = [[[] for _ in range(self.map_width)] for _ in range(self.map_height)]

        # Asteroids: show as A<nutrinium> (only if nutrinium>0)
        for (x, y), asteroid in asteroid_map.items():
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                cell_entities[y][x].append(f"A{asteroid['nutrinium']}")

        # Trading posts
        for post in self.trading_posts:
            if 0 <= post['x'] < self.map_width and 0 <= post['y'] < self.map_height:
                cell_entities[post['y']][post['x']].append("T")

        # Enemies - use ship names (E1, E2, etc.)
        for i, ship in enumerate(self.opponent_ships):
            if not ship.get('destroyed', False):
                x, y = ship['x'], ship['y']
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    # Use ship name instead of index
                    ship_name = ship.get('name', f'E{i+1}')
                    cell_entities[y][x].append(ship_name)

        # Player - use ship name (P)
        if not self.player_ship.get('destroyed', False):
            px, py = self.player_ship['x'], self.player_ship['y']
            if 0 <= px < self.map_width and 0 <= py < self.map_height:
                # Use ship name for player
                player_name = self.player_ship.get('name', 'P')
                cell_entities[py][px].insert(0, player_name)

        # Prepare a bordered grid display. Choose a reasonable cell width.
        # Determine cell width (allow override via self.cell_width)
        if self.cell_width is not None and isinstance(self.cell_width, int) and self.cell_width > 0:
            cell_width = max(3, min(40, int(self.cell_width)))
        else:
            cell_width = max(6, min(12, 80 // max(1, self.map_width)))

        # Determine rendering window (full map or minimap around player)
        if self.minimap_mode and self.player_ship is not None:
            px, py = self.player_ship['x'], self.player_ship['y']
            r = max(0, int(self.minimap_radius))
            x_min = max(0, px - r)
            x_max = min(self.map_width - 1, px + r)
            y_min = max(0, py - r)
            y_max = min(self.map_height - 1, py + r)
        else:
            x_min, x_max = 0, self.map_width - 1
            y_min, y_max = 0, self.map_height - 1

        x_count = x_max - x_min + 1

        # Top header with column indices centered for the rendered window
        header = '     ' + ''.join(str(i % 10).center(cell_width + 3) for i in range(x_min, x_max + 1))
        print(header)

        # Build box-drawing borders so each cell is enclosed (for the window width):
        segment = '-' * (cell_width + 2)
        top_border = '   ' + '+' + '+'.join([segment] * x_count) + '+'
        mid_border = '   ' + '+' + '+'.join([segment] * x_count) + '+'
        bottom_border = '   ' + '+' + '+'.join([segment] * x_count) + '+'

        print(top_border)

        for y in range(y_min, y_max + 1):
            # Build row string with vertical separators
            row_cells = []
            for x in range(x_min, x_max + 1):
                items = cell_entities[y][x]
                if items:
                    cell_text = ','.join(items)
                else:
                    cell_text = ''
                # Truncate if too long
                if len(cell_text) > cell_width:
                    cell_text = cell_text[:cell_width - 1] + '...'
                row_cells.append(cell_text.center(cell_width + 2))

            # Print row with left index and vertical separators
            print(f" {str(y % 10)}  |" + '|'.join(row_cells) + '|')

            # Print middle separator between rows (except after last rendered row)
            if y < y_max:
                print(mid_border)

        # Bottom border
        print(bottom_border)

    def close(self):
        """Clean up resources"""
        pass


# Register the environment with Gymnasium
gym.register(
    id='ProspectorsPirates-v0',
    entry_point='pnp_env:ProspectorsPiratesEnv',
)
