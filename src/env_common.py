"""
Prospectors n Pirates - shared environment foundation.

Centralizes the imports, module logger, action/AI enums and reward classes used by :class:`ProspectorsPiratesEnv` and all of its mixin modules. Every env module does ``from env_common import *`` so the public API (`'ActionType'`, `'OpponentAIType'`, `RewardConfig`, `RewardCalculator` and the re-exported reward/model helpers) stays importable from a single place.
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
from model_adapter import wrap_model_with_compat

# Import model specification system for flexible per-model observation/action spaces
from model_specs import (
    ModelSpec, ObservationSpec, ActionSpec,
    ObservationGenerator, FullObservationGenerator, CompactObservationGenerator, SensorOnlyObservationGenerator,
    get_observation_generator, get_named_model_spec, DEFAULT_FULL_SPEC
)

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
    PLUNDER = 14  # steal nutrinium from a shields-down ship in the same zone
    SALVAGE = 15  # recover nutrinium from wreckage at the current location (module)
    REPAIR = 16  # restore hull at a trading post (module)
    NEGOTIATE = 17  # negotiate a team bonus at the objective trading post
    LOWER_SHIELDS = 18  # drop shields into DRAINING state

class OpponentAIType(IntEnum):
    """AI behavior types for opponent ships"""
    MODEL = 0  # Uses a trained RL model for decision-making
    BOT_V2 = 1  # Delegates to the production heuristic bot (bot_v2.get_action)
    BOT_V3 = 2  # Delegates to the prospector-economy bot (bot_v3.get_action)
    BOT_V4 = 3  # Delegates to the pirate-raider bot (bot_v4.get_action)
    BOT_V5 = 4  # Delegates to the balanced miner-trader bot (bot_v5.get_action)
    BOT_V6 = 5  # Delegates to the model-backed bot (bot_v6.get_action)
    BOT_V7 = 6  # Delegates to the dummy-miner bot (bot_v7.get_action)

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
        ActionType.JUMP_TO_TRADING_POST: 1.0,
        ActionType.RESPAWN: 1.0,
        ActionType.PLUNDER: 1.0,
        ActionType.SALVAGE: 1.0,
        ActionType.REPAIR: 1.0,
        ActionType.NEGOTIATE: 1.0,
ActionType.LOWER_SHIELDS: 0.0,
})
success_bonus: float = 0.0
failure_penalty: float = 0.0
# If True, the environment will construct a RewardCalculatorComposite automatically
use_composite: bool = True
# Optional list of component specifications to include in the composite.
# Each entry can be either a string (component class name) or a dict:
# - {'name': 'DistanceToAsteroidReward', 'params': {'weight': 0.05}}
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