"""Model compatibility and observation adapter utilities.

This module centralizes model compatibility behavior so player and enemy models
can each use the observation-construction logic that matches their trained
observation space (Dict vs flat Box, and varying feature lengths).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseObservationAdapter(ABC):
    """Adapts an environment observation to a model-specific observation format."""

    @abstractmethod
    def adapt(self, observation: Any) -> Any:
        """Return observation adapted for the target model."""


class DictObservationAdapter(BaseObservationAdapter):
    """Adapter for models trained with Dict observation spaces."""

    def __init__(self, target_obs_size: Optional[int]):
        self.target_obs_size = target_obs_size

    def adapt(self, observation: Any) -> Any:
        if not isinstance(observation, dict):
            return observation

        if self.target_obs_size is None or 'observation' not in observation:
            return observation

        obs_arr = observation['observation']
        if hasattr(obs_arr, 'shape') and obs_arr.shape[-1] > self.target_obs_size:
            adapted = dict(observation)
            adapted['observation'] = obs_arr[..., :self.target_obs_size]
            return adapted

        return observation


class FlatObservationAdapter(BaseObservationAdapter):
    """Adapter for models trained with flat Box observation spaces."""

    def __init__(self, target_obs_size: Optional[int]):
        self.target_obs_size = target_obs_size

    def adapt(self, observation: Any) -> Any:
        # New env provides Dict observation; legacy models may expect flat arrays.
        if isinstance(observation, dict):
            observation = observation.get('observation', observation)

        if self.target_obs_size is not None and hasattr(observation, 'shape'):
            if observation.shape[-1] > self.target_obs_size:
                observation = observation[..., :self.target_obs_size]

        return observation


class ObservationAdapterFactory:
    """Builds an observation adapter from a model observation space."""

    @staticmethod
    def from_model_space(model_obs_space: Any) -> BaseObservationAdapter:
        from gymnasium import spaces

        if isinstance(model_obs_space, spaces.Dict):
            obs_size = None
            if 'observation' in model_obs_space.spaces:
                obs_size = model_obs_space['observation'].shape[0]
            return DictObservationAdapter(obs_size)

        if isinstance(model_obs_space, spaces.Box):
            return FlatObservationAdapter(model_obs_space.shape[0])

        return FlatObservationAdapter(None)


class ModelCompatibilityAdapter:
    """Compatibility wrapper for action/observation mismatches and masking."""

    def __init__(
        self,
        model: Any,
        old_action_space_size: int,
        new_action_space_size: int,
        enable_action_masking: bool = True,
    ):
        self.model = model
        self.old_size = int(old_action_space_size)
        self.new_size = int(new_action_space_size)
        self.enable_action_masking = bool(enable_action_masking)

        self.masked_predict_calls = 0
        self.masked_overrides = 0
        self.masked_fallbacks = 0

        self._obs_adapter = ObservationAdapterFactory.from_model_space(
            getattr(self.model, 'observation_space', None)
        )

        self._can_mask_logits = (
            hasattr(self.model, 'policy')
            and hasattr(self.model.policy, 'obs_to_tensor')
            and hasattr(self.model.policy, 'get_distribution')
        )

    def _adapt_observation(self, observation: Any) -> Any:
        return self._obs_adapter.adapt(observation)

    def _remap_action(self, action_int: int) -> int:
        if self.old_size > self.new_size:
            if action_int == 12:
                return 0
            if action_int == 13:
                return 12
            if action_int == 14:
                return 13
        return action_int

    def _masked_argmax(self, obs_for_model: Any, action_mask: Any, deterministic: bool) -> Optional[int]:
        try:
            import torch

            policy = self.model.policy
            policy.set_training_mode(False)
            obs_tensor, _ = policy.obs_to_tensor(obs_for_model)
            with torch.no_grad():
                dist = policy.get_distribution(obs_tensor)
                logits = dist.distribution.logits
        except Exception:
            return None

        n_actions_model = int(logits.shape[-1])
        env_mask = np.asarray(action_mask, dtype=np.int8)

        if self.old_size > self.new_size and n_actions_model == self.old_size:
            model_mask = np.zeros(n_actions_model, dtype=np.int8)
            limit = min(12, env_mask.shape[0])
            model_mask[:limit] = env_mask[:limit]
            if n_actions_model > 12:
                model_mask[12] = 0
            if n_actions_model > 13 and env_mask.shape[0] > 12:
                model_mask[13] = env_mask[12]
            if n_actions_model > 14 and env_mask.shape[0] > 13:
                model_mask[14] = env_mask[13]
        else:
            if env_mask.shape[0] >= n_actions_model:
                model_mask = env_mask[:n_actions_model]
            else:
                model_mask = np.zeros(n_actions_model, dtype=np.int8)
                model_mask[:env_mask.shape[0]] = env_mask

        if model_mask.sum() == 0:
            return None

        import torch

        mask_t = torch.as_tensor(model_mask.astype(bool), device=logits.device).unsqueeze(0)
        masked_logits = logits.masked_fill(~mask_t, float('-inf'))

        if deterministic:
            action_t = masked_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(masked_logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                action_t = masked_logits.argmax(dim=-1)
            else:
                action_t = torch.multinomial(probs, 1).squeeze(-1)

        return int(action_t.cpu().item())

    def predict(self, observation: Any, deterministic: bool = False):
        action_mask = None
        if (
            self.enable_action_masking
            and isinstance(observation, dict)
            and 'action_mask' in observation
        ):
            action_mask = observation['action_mask']

        obs_for_model = self._adapt_observation(observation)

        if action_mask is not None and self._can_mask_logits:
            self.masked_predict_calls += 1
            masked_action_model_space = self._masked_argmax(
                obs_for_model,
                action_mask,
                deterministic,
            )
            if masked_action_model_space is not None:
                raw_action, _ = self.model.predict(obs_for_model, deterministic=True)
                raw_int = int(raw_action.item() if hasattr(raw_action, 'item') else raw_action)
                if raw_int != masked_action_model_space:
                    self.masked_overrides += 1
                action_int = self._remap_action(masked_action_model_space)
                return np.array(action_int), None
            self.masked_fallbacks += 1

        action, state = self.model.predict(obs_for_model, deterministic=deterministic)
        action_int = int(action) if hasattr(action, '__int__') else int(action.item())
        action_int = self._remap_action(action_int)
        return np.array(action_int), state


def wrap_model_with_compat(model: Any, env_action_space_size: int, enable_action_masking: bool = True) -> ModelCompatibilityAdapter:
    """Wrap a raw model in compatibility behavior used by player and enemy paths."""
    model_action_space = getattr(getattr(model, 'action_space', None), 'n', env_action_space_size)
    return ModelCompatibilityAdapter(
        model,
        old_action_space_size=model_action_space,
        new_action_space_size=env_action_space_size,
        enable_action_masking=enable_action_masking,
    )


class MultiDiscreteObsAdapter:
    """Truncating wrapper for native MultiDiscrete models on a larger env obs.

    Native models use a MultiDiscrete action space whose vector output the env
    consumes directly, so the scalar-oriented :class:`ModelCompatibilityAdapter`
    cannot wrap them. When the environment observation grows (e.g. appended
    action-restriction features) but the model was trained on the shorter vector,
    this adapter truncates the observation to the model's expected size and passes
    the model's vector action through unchanged.
    """

    def __init__(self, model: Any):
        self.model = model
        self._obs_adapter = ObservationAdapterFactory.from_model_space(
            getattr(self.model, 'observation_space', None)
        )

    def predict(self, observation: Any, deterministic: bool = False):
        obs_for_model = self._obs_adapter.adapt(observation)
        return self.model.predict(obs_for_model, deterministic=deterministic)


def wrap_multidiscrete_model_with_obs_compat(model: Any) -> MultiDiscreteObsAdapter:
    """Wrap a native MultiDiscrete model so larger env observations are truncated."""
    return MultiDiscreteObsAdapter(model)

