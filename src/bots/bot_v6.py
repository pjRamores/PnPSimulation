"""Prospectors & Pirates bot (v6) -- model-backed.

Unlike the pure-stdlib heuristic bots (``bot_v2`` .. ``bot_v5``), this bot
delegates the decision to a trained reinforcement-learning model. It mirrors the
same 2026 lambda contract -- a single :func:`get_action` that takes an
ActionRequest dict and returns ``{"actionType": str, "payload"?: ...}`` -- but
the action is chosen by running the model over a reconstructed observation.

How it works:

1. :class:`_Context` parses the ActionRequest (``me`` / ``sensors`` /
   ``gameState.metadata``) into the fields the environment's observation builder
   consumes.
2. :func:`_build_observation` reconstructs the environment's 224-dim observation
   vector following the exact layout of ``env_observation_mixin._get_observation``.
   The "global" segments (top asteroids, nearest trading post, enemy extremes,
   nearest wreckage) are reconstructed from SENSOR-VISIBLE entities only -- the
   lambda only sees what the server reports in ``sensors`` (entities within
   sensor range), so these are necessarily local/myopic compared to the
   in-environment AIs that enjoy global map knowledge.
3. :func:`_mask_state` adapts the parsed state into a
   ``utils.action_masker.MaskState`` and the shared action-masking utility
   builds the env's 19-action validity mask (single source of truth with the
   environment) over SENSOR-VISIBLE entities only.
4. The model predicts an action; its raw action TYPE is run through
   ``action_masker.mask_action`` so the bot only ever commits to a valid action
   (the same enforcement the environment applies to the player). The model's own
   target coordinate / energy bin are reused for JUMP / ATTACK payloads.
5. :func:`_action_to_response` converts the chosen env action id back into a
   lambda response dict (the inverse of the environment's ``_translate_bot_action``).

The default model is configurable via :data:`MODEL_NAME` (resolved relative to
``../models/<MODEL_NAME>``). Any failure to import numpy / stable_baselines3 or
to load the model degrades gracefully to ``{"actionType": "WAIT"}``.
"""

import math
import os
import sys
import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _NUMPY_OK = True
except Exception:  # pragma: no cover - numpy is expected but degrade gracefully
    np = None
    _NUMPY_OK = False

# The observation/action-mask reconstruction is shared with the environment (so a
# model trains on byte-identical observations to what it sees here at inference).
# The shared modules live in ``src`` locally (this file's parent's parent); on
# AWS Lambda the ``bots`` folder IS the deployment root, so they sit alongside
# this file. Add BOTH so the shared module resolves in either layout.
try:
    _BOTS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/bots or pkg root
    _SRC_DIR = os.path.dirname(_BOTS_DIR)                   # .../src
    for _dep_dir in (_BOTS_DIR, _SRC_DIR):
        if _dep_dir not in sys.path:
            sys.path.insert(0, _dep_dir)
    from obs_reconstruction import (
        _Context,
        _build_observation,
        _mask_state,
        _get_masker,
        _top_asteroids,
        _entity_at,
        _nearest_entity,
        _ACTION_NAMES,
        _NUM_ACTIONS,
        _ENERGY_BINS,
        _TOP_ASTEROIDS_COUNT,
    )
except Exception:  # pragma: no cover - degrade gracefully if the module is unavailable
    _Context = None
    _build_observation = None
    _mask_state = None
    _get_masker = lambda: None  # noqa: E731
    _top_asteroids = None
    _entity_at = None
    _nearest_entity = None
    _ACTION_NAMES = [
        "WAIT", "MINE", "MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST",
        "RECHARGE", "RECHARGE_END", "ATTACK", "JUMP_TO_ASTEROID", "SELL",
        "RAISE_SHIELDS", "JUMP_TO_TRADING_POST", "RESPAWN", "PLUNDER", "SALVAGE",
        "REPAIR", "NEGOTIATE", "LOWER_SHIELDS",
    ]
    _NUM_ACTIONS = 19
    _ENERGY_BINS = 11
    _TOP_ASTEROIDS_COUNT = 5


# ----------------------------------------------------------------------------
# Configuration (editable)
# ----------------------------------------------------------------------------
# Default model file under ``../models/`` (the ``.zip`` suffix is optional --
# stable_baselines3 appends it). Point this at any current full-spec model
# (224-dim Dict observation, 19 action types). ``ppo_pnp_model_v9`` is a native
# MultiDiscrete([19, map_w, map_h, energy_bins]) model that matches the env.
# MODEL_NAME = "ppo_pnp_model_v5"
MODEL_NAME = "ppo_pnp_model_v7"
# MODEL_NAME = "ppo_checkpoint_13M_v5"

# Observation / action format the loaded model expects, resolved via
# ``model_specs.get_named_model_spec``. Supported names: ``"FULL"`` (224-dim,
# the default), ``"COMPACT"`` (20-dim essentials) and ``"SENSOR_ONLY"``
# (6 + (2*sensor_range+1)^2 local grid). The matching observation is
# reconstructed from sensor-visible entities. Unknown names fall back to FULL.
# MODEL_SPEC = "FULL"
MODEL_SPEC = "COMPACT"

_OBS_SIZE = 224
_DEFAULT_ATTACK_ENERGY = 20     # baseline ATTACK payload when the model gives none

# Simple env action ids that translate to a bare ``{"actionType": NAME}``.
_SIMPLE_ACTIONS = {
    "WAIT", "MINE", "RECHARGE", "RECHARGE_END", "RAISE_SHIELDS",
    "LOWER_SHIELDS", "RESPAWN", "PLUNDER", "SALVAGE", "REPAIR", "NEGOTIATE",
}


def _to_response(payload):
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


# ----------------------------------------------------------------------------
# Lazy model loading (cached at module scope)
# ----------------------------------------------------------------------------
_MODEL = "unset"  # sentinel -> (model, is_multidiscrete) tuple, or None on failure


def _load_model():
    """Load and cache the configured model. Returns ``(model, is_md)`` or ``None``.

    The parent ``PnPSimulation`` directory is added to ``sys.path`` so the model
    and its dependencies resolve. Any failure (missing numpy / sb3 / model file)
    is swallowed and cached as ``None`` so callers fall back to WAIT without
    re-attempting a slow load every tick.
    """
    global _MODEL
    if _MODEL != "unset":
        return _MODEL

    result = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # .../src/bots (or the
                                                           # Lambda package root)
        src_dir = os.path.dirname(here)                    # .../src
        repo_root = os.path.dirname(src_dir)               # .../PnPSimulation

        # Make sibling dependency modules (model_specs, obs_reconstruction,
        # utils, ...) importable in BOTH layouts: locally they live in ``src``
        # (the parent of ``bots``); on AWS Lambda the ``bots`` folder IS the
        # deployment root, so they sit alongside this file in ``here``.
        for dep_dir in (here, src_dir):
            if dep_dir not in sys.path:
                sys.path.insert(0, dep_dir)

        # Models may live next to this file (``<here>/models`` -- the local
        # ``src/bots/models`` AND the Lambda ``bots/models``), or at the repo
        # root (``PnPSimulation/models``). Pick the first candidate that exists
        # (stable_baselines3 appends the ``.zip`` suffix on load).
        model_path = os.path.join(here, "models", MODEL_NAME)
        for candidate in (
            os.path.join(here, "models", MODEL_NAME),
            os.path.join(repo_root, "models", MODEL_NAME),
        ):
            if os.path.exists(candidate) or os.path.exists(candidate + ".zip"):
                model_path = candidate
                break

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
        logger.exception(
            "bot_v6: failed to load model %r (tried %r) -- falling back to WAIT",
            MODEL_NAME, locals().get("model_path", "<unresolved>"),
        )
        result = None

    _MODEL = result
    return result


_SPEC = "unset"  # sentinel -> resolved model_specs.ModelSpec, or None on failure


def _load_spec():
    """Resolve and cache the :data:`MODEL_SPEC` into a ``model_specs.ModelSpec``.

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
        here = os.path.dirname(os.path.abspath(__file__))  # bots dir or Lambda root
        src_dir = os.path.dirname(here)  # .../src
        if dep_dir in (here, src_dir):
            if dep_dir not in sys.path:
                sys.path.insert(0, dep_dir)
        from model_specs import get_named_model_spec, DEFAULT_FULL_SPEC
        result = get_named_model_spec(MODEL_SPEC) or DEFAULT_FULL_SPEC
    except Exception:
        result = None
    _SPEC = result
    return result

# ----------------------------------------------------------------------------
# Model prediction + action-mask enforcement
# ----------------------------------------------------------------------------

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
        return _to_response(get_model_action(action_request))
    except Exception:
        return {"actionType": "WAIT"}