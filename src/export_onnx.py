"""Export a trained SB3 PPO policy to ONNX for lightweight lambda inference.

The lambda (``bot_v6``) must run under a 250MB zip limit, which torch +
stable_baselines3 blow past. This script runs in the training environment (which
has torch + SB3) and emits a portable ``<model>.onnx`` plus a small
``<model>.onnx.meta.json`` sidecar describing the observation/action layout. The
lambda then serves the policy with ``onnxruntime`` + ``numpy`` only.

The exported graph takes the two observation-dict inputs the policy consumes --
``observation`` (float32 [batch, obs_size]) and ``action_mask`` (float32
[batch, num_actions]) -- and returns the deterministic action (``action``):

* single-Discrete model -> int64 [batch]
* MultiDiscrete model -> int64 [batch, k]

Usage:

  python src/export_onnx.py --model src/bots/models/ppo_pnp_model_v5
  python src/export_onnx.py --model models/ppo_checkpoint_16M_v3 --use-masked-ppo
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from torch.distributions import Distribution

Distribution.set_default_validate_args(False)

class _OnnxPolicy(torch.nn.Module):
    def __init__(
            self,
            policy: torch.nn.Module,
            action_dims: list[int],
            use_masked_ppo: bool = False,
    ):
        super().__init__()
        self.policy = policy
        self.action_dims = action_dims
        self.use_masked_ppo = use_masked_ppo

    def forward(self, observation: torch.Tensor, action_mask: torch.Tensor):
        obs = {"observation": observation, "action_mask": action_mask}

        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        logits = self.policy.action_net(latent_pi)

        # 1. Single Discrete Case
        if len(self.action_dims) == 1:
            if self.use_masked_ppo:
                neg_inf = torch.full_like(logits, -1e20)  # Use -1e20 to match SB3 exactly
                logits = torch.where(action_mask > 0, logits, neg_inf)
            return torch.argmax(logits, dim=1)

        # 2. MultiDiscrete Case
        if self.use_masked_ppo:
            # If the mask is smaller than total actions, SB3 fills remaining dimensions with 1s
            total_actions = sum(self.action_dims)
            if action_mask.shape[1] < total_actions:
                extra_mask_size = total_actions - action_mask.shape[1]
                # Match the batch dimension dynamically
                ones_mask = torch.ones((action_mask.shape[0], extra_mask_size), device=action_mask.device, dtype=action_mask.dtype)
                full_mask = torch.cat([action_mask, ones_mask], dim=1)
            else:
                full_mask = action_mask

            # Apply the mask globally using SB3's scaling style before splitting
            neg_inf = torch.full_like(logits, -1e20)
            logits = torch.where(full_mask > 0, logits, neg_inf)

        # Split the masked logits into their respective sub-action sizes
        logit_splits = torch.split(logits, self.action_dims, dim=1)
        actions = [torch.argmax(split, dim=1, keepdim=True) for split in logit_splits]

        return torch.cat(actions, dim=1)

def _resolve_model_path(model_arg: str) -> str:
    if os.path.exists(model_arg):
        return model_arg
    if os.path.exists(model_arg + ".zip"):
        return model_arg
    return model_arg

def _space_shape(space) -> int:
    if getattr(space, "shape", None) is None:
        raise ValueError(f"Space has no shape: {space}")
    if len(space.shape) != 1:
        raise ValueError(f"Expected 1D Box space, got shape={space.shape}")
    return int(space.shape[0])  # <--- Change 'space.shape' to 'space.shape[0]'

def _load_model(model_path: str, use_masked_ppo: bool):
    if use_masked_ppo:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise SystemExit(
                "Maskable PPO requested, but sb3-contrib is not installed. "
                "Install it with: pip install sb3-contrib"
            ) from exc

        return MaskablePPO.load(
            model_path,
            device="cpu",
            custom_objects={
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            },
        )

    return PPO.load(
        model_path,
        device="cpu",
        custom_objects={
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )

def export(model_arg: str, opset: int = 17, use_masked_ppo: bool = False) -> str:
    model_path = _resolve_model_path(model_arg)
    print(f">>>> Loading model: {model_path} (use_masked_ppo={use_masked_ppo})")
    model = _load_model(model_path, use_masked_ppo=use_masked_ppo)

    obs_space = model.observation_space
    if not hasattr(obs_space, "spaces"):
        raise SystemExit(
            f"Expected Dict observation space; got {type(obs_space).__name__}."
        )
    if "observation" not in obs_space.spaces:
        raise SystemExit(
            f"Expected observation key 'observation'; got keys={list(obs_space.spaces.keys())}"
        )
    if "action_mask" not in obs_space.spaces:
        raise SystemExit(
            f"Expected observation key 'action_mask'; got keys={list(obs_space.spaces.keys())}"
        )

    obs_size = _space_shape(obs_space.spaces["observation"])
    mask_size = _space_shape(obs_space.spaces["action_mask"])

    act_space = model.action_space
    is_md = hasattr(act_space, "nvec")
    nvec = [int(v) for v in act_space.nvec] if is_md else None
    num_actions = int(sum(nvec)) if is_md else int(act_space.n)

    print(
        f">>>> obs_size={obs_size} mask_size={mask_size} "
        f"is_md={is_md} nvec={nvec} num_actions={num_actions}"
    )
    print("policy:", type(model.policy))
    print("action_space:", model.action_space)
    print("observation_space:", model.observation_space)
    print("action_net:", model.policy.action_net)

    with torch.no_grad():
        tmp_obs = {
            "observation": torch.zeros((1, obs_size), dtype=torch.float32),
            "action_mask": torch.ones((1, mask_size), dtype=torch.float32),
        }
        features = model.policy.extract_features(tmp_obs)
        latent_pi, _ = model.policy.mlp_extractor(features)
        logits = model.policy.action_net(latent_pi)
        print("logits shape:", tuple(logits.shape))

    action_dims = nvec if is_md else [num_actions]
    wrapper = _OnnxPolicy(
        model.policy,
        action_dims=action_dims,
        use_masked_ppo=use_masked_ppo,
    ).eval()

    example_obs = torch.zeros((1, obs_size), dtype=torch.float32)
    example_mask = torch.ones((1, mask_size), dtype=torch.float32)

    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    onnx_path = base + ".onnx"

    torch.onnx.export(
        wrapper,
        (example_obs, example_mask),
        onnx_path,
        input_names=["observation", "action_mask"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action_mask": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=opset,
    )
    print(f">>>> Wrote ONNX graph: {onnx_path}")

    meta = {
        "obs_size": obs_size,
        "mask_size": mask_size,
        "num_actions": num_actions,
        "is_multidiscrete": bool(is_md),
        "nvec": nvec,
        "input_names": ["observation", "action_mask"],
        "output_name": "action",
        "opset": opset,
        "use_masked_ppo": use_masked_ppo,
    }
    meta_path = onnx_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f">>>> Wrote metadata: {meta_path}")

    _verify_parity(
        model,
        onnx_path,
        obs_size,
        mask_size,
        n=64,
        use_masked_ppo=use_masked_ppo, # <--- Pass the flag here
    )
    return onnx_path

def _verify_parity(
        model,
        onnx_path: str,
        obs_size: int,
        mask_size: int,
        n: int = 64,
        use_masked_ppo: bool = False,
) -> None:
    """Compare ONNX argmax actions vs SB3 predict over random observations."""
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f">>>> Skipping parity check (onnxruntime unavailable): {exc}")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    mismatches = 0

    # Calculate total expected flat action space size for SB3 distribution masking
    is_md = hasattr(model.action_space, "nvec")
    total_actions = int(sum(model.action_space.nvec)) if is_md else int(model.action_space.n)

    for _ in range(n):
        obs = rng.standard_normal((1, obs_size)).astype(np.float32)
        mask = (rng.random((1, mask_size)) > 0.5).astype(np.float32)
        mask[0, 0] = 1.0

        # Pad the mask for SB3's action_masks parameter if it's smaller than total_actions
        if use_masked_ppo and mask.shape[1] < total_actions:
            padding = np.ones((1, total_actions - mask.shape[1]), dtype=np.float32)
            sb3_action_mask = np.concatenate([mask, padding], axis=1)
        else:
            sb3_action_mask = mask

        sb3_action, _ = model.predict(
            {"observation": obs, "action_mask": mask},
            action_masks=sb3_action_mask if use_masked_ppo else None, # <--- Pass the padded mask here
            deterministic=True,
        )
        onnx_action = sess.run(
            ["action"],
            {"observation": obs, "action_mask": mask},
        )

        if not np.array_equal(
                np.asarray(sb3_action).reshape(-1),
                np.asarray(onnx_action).reshape(-1),
        ):
            mismatches += 1

    if mismatches:
        print(f">>>> PARITY WARNING: {mismatches}/{n} actions differ from SB3")
    else:
        print(f">>>> PARITY OK: {n}/{n} actions match SB3 predict")

def main() -> None:
    parser = argparse.ArgumentParser(description="Export SB3 PPO policy to ONNX.")
    parser.add_argument(
        "--model",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bots",
            "models",
            "ppo_pnp_model_v5",
        ),
        help="Path to the SB3 model (with or without .zip).",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--use-masked-ppo",
        action="store_true",
        help="Load/export as sb3_contrib.MaskablePPO and apply action masking in ONNX.",
    )
    args = parser.parse_args()
    export(args.model, opset=args.opset, use_masked_ppo=args.use_masked_ppo)

if __name__ == "__main__":
    main()
