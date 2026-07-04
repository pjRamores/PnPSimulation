"""
Export a trained SB3 PPO policy to ONNX for lightweight lambda inference.

The lambda (`'bot_v6'`) must run under a 250MB zip limit, which torch + stable_baselines3 blow past.
This script runs in the training environment (which has torch + SB3) and emits a portable `<model>.onnx` plus a small
`<model>.onnx.meta.json` sidecar describing the observation/action layout. The lambda then serves the policy with `onnxruntime` + `numpy` only.

The exported graph takes the two observation-dict inputs the policy consumes -- `'observation'` (float32 [1, obs_size]) and `'action_mask'` (float32 [1, num_actions]) -- and returns the deterministic action (`'action'`):

* single-Discrete model -> int64 [1]         (the action id)
* MultiDiscrete model    -> int64 [1, k]     ([action_type, slot, energy_bin])

Usage (from the PnPSimulation venv):
    python src/export_onnx.py --model src/bots/models/ppo_pnp_model_v5

The `.zip` suffix on the model path is optional (SB3 appends it). Outputs are written next to the input model.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO


class _OnnxPolicy(torch.nn.Module):
    """Wrap an SB3 policy so ONNX sees plain tensors, not an obs dict.

    Rebuilds the `{observation`, "action_mask"}` dict the policy expects and returns the deterministic action (the policy's distribution mode).
    """

    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor, action_mask: torch.Tensor):
        obs = {"observation": observation, "action_mask": action_mask}
        return self.policy._predict(obs, deterministic=True)


def _resolve_model_path(model_arg: str) -> str:
    if os.path.exists(model_arg) or model_arg.endswith(".zip"):
        return model_arg
    if os.path.exists(model_arg + ".zip"):
        return model_arg
    return model_arg


def _space_shape(space) -> int:
    return int(space.shape[0])


def export(model_arg: str, opset: int = 17) -> str:
    model_path = _resolve_model_path(model_arg)
    print(f">>> Loading model: {model_path}")
    model = PPO.load(
        model_path,
        device="cpu",
        custom_objects={
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )

    obs_space = model.observation_space
    if not (hasattr(obs_space, "spaces") and "observation" in obs_space.spaces):
        raise SystemExit(
            f"Expected a Dict observation space with an 'observation' key; got {type(obs_space).__name__}."
        )
    obs_size = _space_shape(obs_space.spaces["observation"])
    mask_size = _space_shape(obs_space.spaces["action_mask"])

    act_space = model.action_space
    is_md = hasattr(act_space, "nvec")
    nvec = [int(v) for v in act_space.nvec] if is_md else None
    num_actions = int(nvec[0]) if is_md else int(act_space.n)

    print(
        f">>> obs_size={obs_size} mask_size={mask_size} "
        f"is_md={is_md} nvec={nvec} num_actions={num_actions}"
    )

    wrapper = _OnnxPolicy(model.policy).eval()

    example_obs = torch.zeros(1, obs_size, dtype=torch.float32)
    example_mask = torch.ones(1, mask_size, dtype=torch.float32)

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
}
meta_path = onnx_path + ".meta.json"
with open(meta_path, "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2)
print(f">>>> Wrote metadata: {meta_path}")

_verify_parity(model, onnx_path, obs_size, mask_size, is_md)
return onnx_path

def _verify_parity(model, onnx_path: str, obs_size: int, mask_size: int, is_md: bool, n: int = 64) -> None:
    """Compare ONNX argmax actions vs SB3 predict over random observations."""
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - parity is best-effort
        print(f">>>> Skipping parity check (onnxruntime unavailable): {exc}")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(n):
        obs = rng.standard_normal((1, obs_size)).astype(np.float32)
        mask = (rng.random((1, mask_size)) > 0.5).astype(np.float32)
        mask[0, 0] = 1.0  # keep at least one action valid

        sb3_action, _ = model.predict(
            {"observation": obs, "action_mask": mask}, deterministic=True
        )
        onnx_action = sess.run(
            ["action"], {"observation": obs, "action_mask": mask}
        )[0]

        if not np.array_equal(
                np.asarray(sb3_action).reshape(-1), np.asarray(onnx_action).reshape(-1)
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
            "bots", "models", "ppo_pnp_model_v5",
        ),
        help="Path to the SB3 model (with or without .zip).",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.model, opset=args.opset)

if __name__ == "__main__":
    main()