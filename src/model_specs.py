"""
Model Specification System

Defines extensible observation and action space specifications that allow
different models to use different observation/action formats while competing
in the same environment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import IntEnum
import numpy as np
from abc import ABC, abstractmethod
import copy


class ActionSpec:
    """Specification for a model's action space."""

    def __init__(self,
                 action_space_size: int = 19,
                 action_mapping: Optional[Dict[int, int]] = None,
                 description: str = "Standard 19 actions"):
        """
        Args:
            action_space_size: Number of actions the model outputs
            action_mapping: Map model actions to environment actions (None = identity mapping)
            description: Human-readable description
        """
        self.action_space_size = action_space_size
        self.action_mapping = action_mapping or {i: i for i in range(action_space_size)}
        self.description = description

    def map_action(self, model_action: int, env_action_space_size: int = 19) -> int:
        """Map a model action to an environment action."""
        if model_action not in self.action_mapping:
            return 0  # Default to WAIT if unmapped
        return self.action_mapping[model_action]

    def to_dict(self) -> dict:
        return {
            'action_space_size': self.action_space_size,
            'action_mapping': self.action_mapping,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, spec_dict: dict) -> 'ActionSpec':
        return cls(**spec_dict)


class ObservationSpec:
    """Specification for a model's observation space."""

    def __init__(self,
                 observation_type: str = 'full',
                 feature_names: Optional[List[str]] = None,
                 observation_size: Optional[int] = None,
                 include_action_mask: bool = True,
                 description: str = "Standard observation"):
        """
        Args:
            observation_type: Type of observation ('full', 'compact', 'sensor_only', 'custom')
            feature_names: Names of features in observation
            observation_size: Expected size of flattened observation (None = auto-calculated)
            include_action_mask: Whether to include action mask in observation
            description: Human-readable description
        """
        self.observation_type = observation_type
        self.feature_names = feature_names or []
        self.observation_size = observation_size
        self.include_action_mask = include_action_mask
        self.description = description

    def to_dict(self) -> dict:
        return {
            'observation_type': self.observation_type,
            'feature_names': self.feature_names,
            'observation_size': self.observation_size,
            'include_action_mask': self.include_action_mask,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, spec_dict: dict) -> 'ObservationSpec':
        return cls(**spec_dict)


class ModelSpec:
    """Complete specification for a model's I/O format."""

    def __init__(self,
                 name: str = "default",
                 observation_spec: Optional[ObservationSpec] = None,
                 action_spec: Optional[ActionSpec] = None,
                 version: str = "1.0"):
        """
        Args:
            name: Model name/identifier
            observation_spec: Observation specification (default = standard full obs)
            action_spec: Action specification (default = 14 actions)
            version: Spec version
        """
        self.name = name
        self.observation_spec = observation_spec or ObservationSpec('full')
        self.action_spec = action_spec or ActionSpec(19)
        self.version = version

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'observation_spec': self.observation_spec.to_dict(),
            'action_spec': self.action_spec.to_dict(),
            'version': self.version,
        }

    @classmethod
    def from_dict(cls, spec_dict: dict) -> 'ModelSpec':
        obs_spec = ObservationSpec.from_dict(spec_dict['observation_spec']) if 'observation_spec' in spec_dict else None
        act_spec = ActionSpec.from_dict(spec_dict['action_spec']) if 'action_spec' in spec_dict else None
        return cls(
            name=spec_dict.get('name', 'default'),
            observation_spec=obs_spec,
            action_spec=act_spec,
            version=spec_dict.get('version', '1.0'),
        )


class ObservationGenerator(ABC):
    """Base class for generating observations in different formats."""

    def __init__(self, env: Any, spec: ObservationSpec):
        """
        Args:
            env: Reference to the environment
            spec: ObservationSpec defining the format
        """
        self.env = env
        self.spec = spec

    @abstractmethod
    def generate(self, ship: dict) -> Dict[str, np.ndarray]:
        """
        Generate observation for a ship.

        Args:
            ship: Ship dictionary

        Returns:
            Dict with 'observation' and optionally 'action_mask'
        """
        pass

    def _get_action_mask(self, ship: dict, is_player: bool = True) -> np.ndarray:
        """Get action mask for a ship."""
        num_actions = getattr(self.env, 'num_action_types', None)
        if num_actions is None:
            num_actions = getattr(self.env.action_space, 'n', None)
        mask = np.zeros(num_actions, dtype=np.int8)
        for action in range(num_actions):
            is_valid, _ = self.env._is_action_valid_for_state(action, ship, is_player=is_player)
            if is_valid:
                mask[action] = 1
        return mask


class FullObservationGenerator(ObservationGenerator):
    """Generates the full standard observation (current default behavior)."""

    def generate(self, ship: dict) -> Dict[str, np.ndarray]:
        """Generate full observation format using the legacy implementation."""
        # Set a flag to prevent infinite recursion when using use_spec=True
        original_flag = getattr(self.env, '_generating_observation', False)
        self.env._generating_observation = True
        try:
            # Get the legacy full observation without using the spec system
            obs = self.env._get_observation(
                skip_mask=(not self.spec.include_action_mask),
                use_spec=False,
            )
            if not self.spec.include_action_mask and isinstance(obs, dict) and 'action_mask' in obs:
                obs = {'observation': obs['observation']}
            return obs
        finally:
            self.env._generating_observation = original_flag


class CompactObservationGenerator(ObservationGenerator):
    """Generates a compact observation (essential features only)."""

    def generate(self, ship: dict) -> Dict[str, np.ndarray]:
        """Generate compact observation with just essential features."""
        features = []

        # Ship state (8 values)
        features.extend([
            ship['x'] / self.env.map_width,
            ship['y'] / self.env.map_height,
            ship['energy'] / self.env.config['max_energy'],
            ship['health'] / self.env.config['max_health'],
            ship['nutrinium'] / self.env.config['max_nutrinium_cargo'],
            ship['credits'] / self.env.config['max_credits'],
            1.0 if ship.get('recharging', False) else 0.0,
            1.0 if ship.get('shields_up', False) else 0.0,
        ])

        # Nearest asteroid (4 values)
        nearest_ast = self.env._get_nearest_entity(ship['x'], ship['y'], self.env.asteroids)
        if nearest_ast:
            dist = self.env._calculate_distance(ship['x'], ship['y'], nearest_ast['x'], nearest_ast['y'])
            features.extend([
                nearest_ast['x'] / self.env.map_width,
                nearest_ast['y'] / self.env.map_height,
                nearest_ast['nutrinium'] / max(nearest_ast['mass'], 1),
                dist / (self.env.map_width + self.env.map_height),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 1.0])

        # Nearest trading post (3 values)
        nearest_post = self.env._get_nearest_entity(ship['x'], ship['y'], self.env.trading_posts)
        if nearest_post:
            dist = self.env._calculate_distance(ship['x'], ship['y'], nearest_post['x'], nearest_post['y'])
            features.extend([
                nearest_post['x'] / self.env.map_width,
                nearest_post['y'] / self.env.map_height,
                dist / (self.env.map_width + self.env.map_height),
            ])
        else:
            features.extend([0.0, 0.0, 1.0])

        # Nearest enemy (5 values)
        nearest_enemy = None
        for enemy in self.env.opponent_ships:
            if not enemy.get('destroyed', False):
                nearest_enemy = enemy
                break

        if nearest_enemy:
            dist = self.env._calculate_distance(ship['x'], ship['y'], nearest_enemy['x'], nearest_enemy['y'])
            features.extend([
                nearest_enemy['x'] / self.env.map_width,
                nearest_enemy['y'] / self.env.map_height,
                nearest_enemy['health'] / self.env.config['max_health'],
                nearest_enemy['nutrinium'] / self.env.config['max_nutrinium_cargo'],
                dist / (self.env.map_width + self.env.map_height),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 1.0])

        obs_array = np.array(features, dtype=np.float32)

        is_player = (ship == self.env.player_ship)
        result = {
            'observation': obs_array,
        }

        if self.spec.include_action_mask:
            result['action_mask'] = self._get_action_mask(ship, is_player=is_player)

        return result


class SensorOnlyObservationGenerator(ObservationGenerator):
    """Generates observation from local sensor grid only (no global context)."""

    def generate(self, ship: dict) -> Dict[str, np.ndarray]:
        """Generate sensor-only observation."""
        features = []

        # Ship essentials (6 values)
        features.extend([
            ship['x'] / self.env.map_width,
            ship['y'] / self.env.map_height,
            ship['energy'] / self.env.config['max_energy'],
            ship['health'] / self.env.config['max_health'],
            ship['nutrinium'] / self.env.config['max_nutrinium_cargo'],
            ship['credits'] / self.env.config['max_credits'],
        ])

        # Local sensor grid (11x11 = 121 values)
        sensor_range = self.env.config['sensor_range']
        grid_size = (2 * sensor_range + 1) ** 2
        sensor_grid = np.zeros(grid_size, dtype=np.float32)

        for i in range(-sensor_range, sensor_range + 1):
            for j in range(-sensor_range, sensor_range + 1):
                gx, gy = ship['x'] + i, ship['y'] + j
                if 0 <= gx < self.env.map_width and 0 <= gy < self.env.map_height:
                    ast = self.env._get_entity_at_location(gx, gy, self.env.asteroids)
                    post = self.env._get_entity_at_location(gx, gy, self.env.trading_posts)
                    enemy = None
                    for e in self.env.opponent_ships:
                        if e['x'] == gx and e['y'] == gy:
                            enemy = e
                            break

                    idx = (i + sensor_range) * (2 * sensor_range + 1) + (j + sensor_range)
                    if enemy:
                        sensor_grid[idx] = 0.9
                    elif post:
                        sensor_grid[idx] = 0.7
                    elif ast:
                        sensor_grid[idx] = 0.5

        features.extend(sensor_grid)

        obs_array = np.array(features, dtype=np.float32)

        is_player = (ship == self.env.player_ship)
        result = {
            'observation': obs_array,
        }

        if self.spec.include_action_mask:
            result['action_mask'] = self._get_action_mask(ship, is_player=is_player)

        return result


# Registry of built-in generators
OBSERVATION_GENERATORS = {
    'full': FullObservationGenerator,
    'compact': CompactObservationGenerator,
    'sensor_only': SensorOnlyObservationGenerator,
}


def get_observation_generator(spec: ObservationSpec, env: Any) -> ObservationGenerator:
    """Get appropriate observation generator for a spec."""
    gen_class = OBSERVATION_GENERATORS.get(spec.observation_type, FullObservationGenerator)
    return gen_class(env, spec)


# Preset specs for common configurations
DEFAULT_FULL_SPEC = ModelSpec(
    name="default_full",
    observation_spec=ObservationSpec('full', description="Full observation with all features"),
    action_spec=ActionSpec(19, description="Standard 19 actions"),
)

DEFAULT_COMPACT_SPEC = ModelSpec(
    name="default_compact",
    observation_spec=ObservationSpec('compact', description="Compact observation with essential features"),
    action_spec=ActionSpec(19, description="Standard 19 actions"),
)

DEFAULT_SENSOR_SPEC = ModelSpec(
    name="default_sensor",
    observation_spec=ObservationSpec('sensor_only', description="Sensor-only observation"),
    action_spec=ActionSpec(19, description="Standard 19 actions"),
)


PRESET_MODEL_SPECS = {
    'DEFAULT_FULL_SPEC': DEFAULT_FULL_SPEC,
    'DEFAULT_COMPACT_SPEC': DEFAULT_COMPACT_SPEC,
    'DEFAULT_SENSOR_SPEC': DEFAULT_SENSOR_SPEC,
    # Friendly aliases
    'FULL': DEFAULT_FULL_SPEC,
    'COMPACT': DEFAULT_COMPACT_SPEC,
    'SENSOR_ONLY': DEFAULT_SENSOR_SPEC,
}


def get_named_model_spec(spec_name: str) -> Optional[ModelSpec]:
    """Resolve a model spec name to a ModelSpec instance.

    Returns a deep copy so callers can safely customize without mutating presets.
    """
    if not spec_name:
        return None
    preset = PRESET_MODEL_SPECS.get(str(spec_name).strip().upper())
    if preset is None:
        return None
    return copy.deepcopy(preset)
