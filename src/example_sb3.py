"""
Example: Using Stable Baselines3 for training RL agents

This example shows how to use the popular Stable Baselines3 library
with the Prospectors n Pirates environment.

Install with: pip install stable-baselines3[extra]
"""

try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    import matplotlib.pyplot as plt

    SB3_AVAILABLE = True
except ImportError:
    print("Stable Baselines3 not installed. Install with: pip install stable-baselines3[extra]")
    SB3_AVAILABLE = False

try:
    from setproctitle import setproctitle

    SETPROCTITLE_AVAILABLE = True
except ImportError:
    SETPROCTITLE_AVAILABLE = False

# Windows-specific: Set console window title for better visibility
import platform

if platform.system() == 'Windows':
    import ctypes


    def set_console_title(title):
        """Set the console window title on Windows"""
        try:
            ctypes.windll.kernel32.SetConsoleTitleW(title)
        except:
            pass
else:
    def set_console_title(title):
        """Placeholder for non-Windows systems"""
        pass

from pnp_env import ProspectorsPiratesEnv, RewardConfig
from model_specs import DEFAULT_FULL_SPEC, get_named_model_spec
import gymnasium as gym
import numpy as np
import os
import csv
from datetime import datetime
from typing import Optional, List
import random


class FlattenDictObsWrapper(gym.ObservationWrapper):
    """Wrapper that flattens a Dict observation space to a flat Box.

    The PnP environment returns ``{'observation': ..., 'action_mask': ...}``.
    Regular PPO / DQN expect a flat ``Box`` observation.  This wrapper extracts
    only the ``'observation'`` key so that models trained with ``MlpPolicy``
    can work.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Replace Dict obs space with the inner Box
        assert isinstance(env.observation_space, gym.spaces.Dict), \
            "FlattenDictObsWrapper requires a Dict observation space"
        self.observation_space = env.observation_space['observation']

    def observation(self, observation):
        if isinstance(observation, dict):
            return observation['observation']
        return observation


class DynamicOpponentsWrapper(gym.Wrapper):
    """Wrapper that samples a new opponent count on each episode reset.

    This allows the agent to train against varying numbers of opponents,
    learning strategies that generalize across different competition levels.

    The wrapper dynamically adjusts the environment's num_opponents and
    recreates the opponent_ships list on each reset.
    """

    def __init__(self, env: gym.Env, min_opponents: int, max_opponents: int):
        """
        Args:
            env: The base environment
            min_opponents: Minimum number of opponents (inclusive)
            max_opponents: Maximum number of opponents (inclusive)
        """
        super().__init__(env)
        self.min_opponents = min_opponents
        self.max_opponents = max_opponents

    def reset(self, **kwargs):
        # Sample new opponent count for this episode
        sampled_count = random.randint(self.min_opponents, self.max_opponents)

        # Update the environment's opponent count
        # This must happen BEFORE calling env.reset() so the reset logic uses it
        self.env.num_opponents = sampled_count

        # Now call the underlying reset which will use the new opponent count
        return self.env.reset(**kwargs)


class ActionMaskWrapper(gym.Wrapper):
    """Wrapper that enforces action masking for standard PPO.

    Standard PPO ignores the action_mask in the observation dict.  This wrapper
    intercepts invalid actions in `step()`, replaces them with a
    randomly-sampled *valid* action, and applies a penalty so the model learns
    to avoid masked actions.

    The penalty-based approach is less efficient than true masked sampling,
    but is significantly better than the silent enforcement
    fallback that existed before, because:
        - The model receives a consistent negative reward for invalid choices.
        - The replacement action is random among valid ones (no bias toward a
          single fallback), so the model can't game the fallback.
    """

    INVALID_ACTION_PENALTY = 0.0  # No penalty -- action is silently replaced with a valid one

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_action_mask = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict) and 'action_mask' in obs:
            self._last_action_mask = obs['action_mask'].copy()
        return obs, info

    def step(self, action):
        # The env action space is MultiDiscrete
        # ([action_type, target_x, target_y, energy_bin]); the action mask applies
        # only to the action_type component (index 0). Support both array actions
        # and bare scalar actions for backward compatibility.
        if self._last_action_mask is not None:
            mask = self._last_action_mask
            action_arr = np.asarray(action)
            is_vector = action_arr.ndim >= 1 and action_arr.size > 1
            action_type = int(action_arr.flat[0]) if action_arr.size else int(action_arr)

            if 0 <= action_type < len(mask) and mask[action_type] == 0:
                # Action type is masked -- pick a random valid action type instead
                valid_actions = [i for i in range(len(mask)) if mask[i] == 1]
                if valid_actions:
                    new_type = int(np.random.choice(valid_actions))
                    if is_vector:
                        action = action_arr.copy()
                        action[0] = new_type
                    else:
                        action = new_type
                # Apply penalty; will be added after step()
                obs, reward, terminated, truncated, info = self.env.step(action)
                reward += self.INVALID_ACTION_PENALTY
                info['action_mask_enforced'] = True
            else:
                obs, reward, terminated, truncated, info = self.env.step(action)
                info['action_mask_enforced'] = False
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            info['action_mask_enforced'] = False

        # Update stored mask for next step
        if isinstance(obs, dict) and 'action_mask' in obs:
            self._last_action_mask = obs['action_mask'].copy()

        return obs, reward, terminated, truncated, info


# Try to import torch for CPU/thread control
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_cpu_mode(efficiency_mode=False, num_threads=None):
    """
    Set CPU usage mode for training.

    Args:
        efficiency_mode: If True, limit CPU usage for efficiency. If False, maximize CPU usage.
        num_threads: Specific number of threads to use (overrides efficiency_mode if set)

    Returns:
        dict with applied settings
    """
    settings = {}

    # Get CPU count
    cpu_count = os.cpu_count() or 4

    # Determine thread count
    if num_threads is not None:
        threads = max(1, min(num_threads, cpu_count))
        settings['mode'] = 'custom'
    elif efficiency_mode:
        # Efficiency mode: Use 50% of CPUs, minimum 1, maximum 4
        threads = max(1, min(4, cpu_count // 2))
        settings['mode'] = 'efficiency'
    else:
        # Performance mode: Use all available CPUs
        threads = cpu_count
        settings['mode'] = 'performance'

    settings['threads'] = threads
    settings['cpu_count'] = cpu_count

    # Set thread counts for PyTorch (only if not already set)
    if TORCH_AVAILABLE:
        try:
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(threads)
            settings['torch_configured'] = True
        except RuntimeError:
            # Threads already set, can't change them
            settings['torch_configured'] = False
            settings['torch_note'] = 'already_configured'
    else:
        settings['torch_configured'] = False

    # Set environment variables for other libraries
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(threads)

    # Windows-specific: Set process priority
    if platform.system() == 'Windows':
        try:
            import psutil
            p = psutil.Process()
            if efficiency_mode:
                # Below normal priority for efficiency mode
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                settings['process_priority'] = 'below_normal'
            else:
                # High priority for performance mode
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                settings['process_priority'] = 'high'
        except ImportError:
            settings['priority_set'] = False
        except Exception as e:
            settings['priority_set'] = False
            settings['priority_error'] = str(e)
    else:
        # Unix-like systems: use nice values
        try:
            import psutil
            p = psutil.Process()
            if efficiency_mode:
                p.nice(10)  # Lower priority (higher nice value)
                settings['process_priority'] = 'nice_10'
            else:
                p.nice(-5)  # Higher priority (lower nice value, requires permissions)
                settings['process_priority'] = 'nice_-5'
        except:
            settings['priority_set'] = False

    return settings


def _ask_to_evaluate(prompt="Press ENTER or SPACE to evaluate the model, or Q to quit: ") -> bool:
    """Prompt the user to continue to evaluation or quit.

    Returns True to continue (evaluate), False to quit.
    Accepts ENTER or SPACE to continue, 'q' or 'Q' to quit. Works on Windows and falls back to input() on other platforms.
    """
    if platform.system() == 'Windows':
        try:
            import msvcrt
            print(prompt, end='', flush=True)
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    # Enter (CR or LF) or Space
                    if key in (b'\r', b'\n', b' '):
                        print('')
                        return True
                    if key.lower() == b'q':
                        print('')
                        return False
                    # Ignore other keys
        except Exception:
            pass  # Fall through to input() below

    # Fallback: use input(); empty line or single space continues, 'q' quits
    try:
        resp = input(prompt)
    except Exception:
        # In non-interactive environments (e.g. test runners), default to continue
        return True
    if resp.strip().lower() == 'q':
        return False
    return True


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress"""

    def __init__(self, verbose=0, max_episodes_to_track=10000, total_timesteps=100000):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_credits = []
        self.episode_lengths = []
        self.max_episodes_to_track = max_episodes_to_track
        self.total_timesteps = total_timesteps
        self.last_reported_progress = 0

    def _on_step(self) -> bool:
        try:
            # Report progress at 10% intervals based on timesteps
            current_progress = (self.num_timesteps / self.total_timesteps) * 100
            progress_milestone = int(current_progress / 10) * 10

            if progress_milestone > self.last_reported_progress and progress_milestone % 10 == 0:
                self.last_reported_progress = progress_milestone
                avg_credits = np.mean(self.episode_credits[-10:]) \
                    if len(self.episode_credits) >= 10 else (
                    np.mean(self.episode_credits) if self.episode_credits else 0)
                print(f"\n[progress_milestone] Progress: {self.num_timesteps}/{self.total_timesteps} timesteps | "
                      f"Episodes: {len(self.episode_credits)} | Avg Credits (last 10): {avg_credits:.1f}")

            # Check if episode is done
            dones = self.locals.get('dones')
            if dones is not None and len(dones) > 0 and dones[0]:
                infos = self.locals.get('infos')
                if infos is not None and len(infos) > 0:
                    info = infos[0]
                    rewards = self.locals.get('rewards')

                    # Safely get reward
                    reward = rewards[0] if rewards is not None and len(rewards) > 0 else 0

                    self.episode_rewards.append(reward)
                    self.episode_credits.append(info.get('player_credits', 0))

                    # Limit memory usage by keeping only recent episodes
                    if len(self.episode_credits) > self.max_episodes_to_track:
                        self.episode_rewards = self.episode_rewards[-self.max_episodes_to_track:]
                        self.episode_credits = self.episode_credits[-self.max_episodes_to_track:]

        except Exception as e:
            # Don't crash training on callback error
            if self.verbose > 0:
                print(f"Warning in callback: {e}")

        return True


def _safe_model_save(model, path):
    """Save a model, working around platform.platform() crash in restricted environments.
    
    SB3's model.save() internally calls platform.platform() to log system info.
    On some corporate/restricted Windows environments this subprocess call fails.
    We monkey-patch platform.platform temporarily to return a safe string.
    """
    import platform as _platform
    _original = _platform.platform
    try:
        _platform.platform = lambda *a, **kw: "Windows"
        model.save(path)
    finally:
        _platform.platform = _original


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints at regular timestep intervals"""

    def __init__(self, save_freq: int, save_path: str, algorithm: str, name_prefix: str = 'checkpoint',
                 verbose: int = 0):
        """
        Args:
            save_freq: Save checkpoint every save_freq timesteps (e.g., 10_000_000 for 10M)
            save_path: Directory to save checkpoints
            algorithm: Algorithm name (for version numbering)
            name_prefix: Prefix for checkpoint files
            verbose: Verbosity level
        """
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.algorithm = algorithm
        self.name_prefix = name_prefix
        self.last_save_timestep = 0

    def _on_step(self) -> bool:
        # Check if we've reached a checkpoint
        if self.num_timesteps - self.last_save_timestep >= self.save_freq:
            # Calculate checkpoint number (e.g., timestep 10M -> checkpoint 1, 20M -> checkpoint 2)
            checkpoint_num = self.num_timesteps // self.save_freq

            # Get version number for this checkpoint
            version = _get_next_version_number(self.algorithm, self.save_path, is_transfer=False)

            # Create checkpoint path
            checkpoint_path = os.path.join(
                self.save_path,
                f'{self.algorithm.lower()}_{self.name_prefix}_{checkpoint_num * self.save_freq // 1_000_000}M_v{version}'
            )

            # Save the model
            _safe_model_save(self.model, checkpoint_path)

            if self.verbose > 0:
                print(f"\n{'=' * 70}")
                print(f"CHECKPOINT SAVED")
                print(f"  Timesteps: {self.num_timesteps:,}")
                print(f"  Path: {checkpoint_path}")
                print(f"  Checkpoint: {checkpoint_num * self.save_freq // 1_000_000}M timesteps")
                print(f"{'=' * 70}\n")

            self.last_save_timestep = self.num_timesteps

        return True


def _get_next_version_number(algorithm, save_path='models/', is_transfer=False):
    """
    Get the next version number for the model.

    Args:
        algorithm: Algorithm name (PPO, DQN, A2C)
        save_path: Directory where models are saved
        is_transfer: If True, increment from existing versions

    Returns:
        Version number (int)
    """
    import re

    if not os.path.exists(save_path):
        return 1

    # Pattern: algorithm_pnp_model_v[number].zip
    pattern = re.compile(rf'{algorithm.lower()}_pnp_model_v(\d+)\.zip', re.IGNORECASE)

    max_version = 0
    for filename in os.listdir(save_path):
        match = pattern.match(filename)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)

    # Increment from max version or start at 1
    return max_version + 1 if max_version > 0 else 1


def _freeze_early_layers(model, algorithm):
    """
    Freeze early layers of the neural network for transfer learning.
    This preserves learned feature representations while allowing
    the policy head to adapt to new tasks.
    """
    try:
        import torch

        if algorithm in ['PPO', 'A2C']:
            # Freeze feature extractor layers
            if hasattr(model.policy, 'mlp_extractor'):
                # Freeze first half of policy network
                policy_net = model.policy.mlp_extractor.policy_net
                num_layers = len(list(policy_net.children()))
                freeze_until = num_layers // 2

                for i, layer in enumerate(policy_net.children()):
                    if i < freeze_until:
                        for param in layer.parameters():
                            param.requires_grad = False

                # Also freeze part of value network
                value_net = model.policy.mlp_extractor.value_net
                for i, layer in enumerate(value_net.children()):
                    if i < freeze_until:
                        for param in layer.parameters():
                            param.requires_grad = False

        elif algorithm == 'DQN':
            # Freeze feature extractor in Q-network
            if hasattr(model.policy, 'q_net'):
                q_net = model.policy.q_net
                if hasattr(q_net, 'features_extractor'):
                    for param in q_net.features_extractor.parameters():
                        param.requires_grad = False

    except Exception as e:
        print(f" Warning: Could not freeze layers: {e}")


def _print_performance_participant_stats(stats_list, opponents):
    """Print participant stats from performance testing in a compact table."""
    if not stats_list:
        return

    from pnp_env import OpponentAIType

    num_episodes = len(stats_list)
    participants = {}

    # Aggregate stats across all episodes
    for stats in stats_list:
        episode_rankings = []
        episode_rankings.append((
            'PLAYER',
            stats.get('player_credits', 0) or 0,
            stats.get('player_nutrinium', 0) or 0,
            stats.get('player_kills', 0) or 0,
            stats.get('player_energy', 0) or 0,
        ))
        
        for enemy in stats.get('enemy_details', []):
            e_name = enemy.get('name', 'ENEMY')
            e_credits = enemy.get('credits', 0) or 0
            e_nutrinium = enemy.get('nutrinium', 0) or 0
            e_kills = enemy.get('kills', 0) or 0
            e_energy = enemy.get('energy', 0) or 0
            episode_rankings.append((e_name, e_credits, e_nutrinium, e_kills, e_energy))

        episode_rankings.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))
        placements = {name: rank + 1 for rank, (name, *_) in enumerate(episode_rankings)}

        # Accumulate stats for PLAYER
        if 'PLAYER' not in participants:
            participants['PLAYER'] = {
                'name': 'PLAYER', 'role': 'PLAYER (new model)',
                'total_credits': 0, 'max_credits': 0,
                'survived': 0,  'episodes': 0,
                'placements': {},
            }
        p = participants['PLAYER']
        p['episodes'] += 1
        p_credits = stats.get('player_credits', 0) or 0
        p['total_credits'] += p_credits
        p['max_credits'] = max(p['max_credits'], p_credits)
        if not stats.get('player_destroyed', False):
            p['survived'] += 1
        placement = placements.get('PLAYER', 0)
        p['placements'][placement] = p['placements'].get(placement, 0) + 1

        # Accumulate stats for enemies
        for i, enemy in enumerate(stats.get('enemy_details', [])):
            e_name = enemy.get('name', 'ENEMY')
            ai_type = enemy.get('ai_type', None)

            # Determine role from opponent list
            if i < len(opponents):
                opp_spec = opponents[i]
                opp_upper = opp_spec.upper()
                if opp_upper in OpponentAIType.__members__ and opp_upper != 'MODEL':
                    role = opp_upper
                else:
                    # Model path
                    role = f"MODEL:{os.path.basename(opp_spec)}"
            else:
                role = 'UNKNOWN'

            if e_name not in participants:
                participants[e_name] = {
                    'name': e_name, 'role': role,
                    'total_credits': 0, 'max_credits': 0,
                    'survived': 0,  'episodes': 0,
                    'placements': {},
                }
            e = participants[e_name]
            e['episodes'] += 1
            p_credits = enemy.get('credits', 0) or 0
            e['total_credits'] += p_credits
            e['max_credits'] = max(e['max_credits'], p_credits)
            if not enemy.get('destroyed', False):
                e['survived'] += 1
            placement = placements.get(e_name, 0)
            e['placements'][placement] = e['placements'].get(placement, 0) + 1

    # Build rows
    rows = []
    for key, p in participants.items():
        eps = p['episodes']
        avg_credits = p['total_credits'] / eps if eps else 0
        survival_rate = (p['survived'] / eps * 100) if eps else 0
        first_place = p['placements'].get(1, 0)
        second_place = p['placements'].get(2, 0)
        third_place = p['placements'].get(3, 0)

        rows.append({
            'name': p['name'],
            'role': p['role'],
            'avg_credits': avg_credits,
            'max_credits': p['max_credits'],
            'survival_rate': survival_rate,
            'first_place': first_place,
            'second_place': second_place,
            'third_place': third_place,
        })

    # Sort by placement priority
    rows.sort(key=lambda r: (
        -r['first_place'],
        -r['second_place'],
        -r['third_place'],
        -r['avg_credits']
    ))

    # Format and print
    ranked_rows = []
    for i, row in enumerate(rows):
        ranked_rows.append((
            str(i + 1),
            row['name'],
            row['role'][0:25],  # Truncate long role names
            f"{row['avg_credits']:.1f}",
            str(row['max_credits']),
            f"{row['survival_rate']:.0f}%",
            str(row['first_place']),
            str(row['second_place']),
            str(row['third_place'])
        ))

    headers = ["Rank", "Ship", "Role", "Avg Cr", "Max Cr", "Surv%", "1st", "2nd", "3rd"]
    all_rows = [headers] + ranked_rows
    cols = list(zip(*all_rows))
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    header_line = "".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "".join('-' * w for w in widths)
    print(f"  {header_line}")
    print(f"  {sep_line}")
    for row in ranked_rows:  # Show all participants
        line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
        print(f"  {line}")


def _test_model_performance(model_path, algorithm, num_episodes=100,
                            map_width=10, map_height=10, max_steps=300):
    """
    Test the newly trained model's performance against existing models.

    Runs a simulation with the new model as PLAYER against opponents including
    existing models from enemy_models.config plus algorithmic AIs.
    
    Args:
        model_path: Path to the newly trained model (without .zip extension)
        algorithm: Algorithm name (PPO, DQN, A2C)
        num_episodes: Number of episodes to simulate (default: 100)
        map_width: Width of the game map (must match the training map so the
                   simulation env's MultiDiscrete action space matches the model)
        map_height: Height of the game map (see `map_width`)
        max_steps: Maximum steps per simulated episode

    Returns:
        dict with keys: 'first_place', 'second_place', 'third_place' (counts)
        Returns None if simulation fails
    """
    try:
        print(f"\n{'=' * 70}")
        print("PERFORMANCE TESTING: Simulating against existing models")
        print(f"{'=' * 70}")
        print(f"Model: {model_path}")
        print(f"Episodes: {num_episodes}")

        # Import GameSimulator
        try:
            from game_simulator import GameSimulator
        except ImportError:
            print("Could not import GameSimulator - skipping performance test")
            return None

        # Load enemy models from config
        enemy_models_config = 'enemy_models.config'
        enemy_models = []

        if os.path.exists(enemy_models_config):
            with open(enemy_models_config, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        enemy_models.append(line)

        if not enemy_models:
            print("No enemy models found in enemy_models.config - using only algorithmic AIs")
        else:
            print(f"Loaded {len(enemy_models)} enemy models from config")

        # Build opponent list: 2x of every algorithmic AI in OpponentAIType, then
        # enemy models. MODEL is excluded from the fixed list because those
        # opponents are supplied by enemy_models (model paths) loaded above.
        from pnp_env import OpponentAIType

        _excluded = (
            OpponentAIType.MODEL,
            OpponentAIType.BOT_V6,
            OpponentAIType.BOT_V8,
        )
        algorithmic_ai_types = [
            ai_type.name
            for ai_type in OpponentAIType
            if ai_type not in _excluded
        ]
        opponents = [
                        name
                        for name in algorithmic_ai_types
                        for _ in range(2)
                    ] + enemy_models

        print(f"  Opponents ({len(opponents)}): {', '.join(opponents)}")

        # Create simulator. The map dimensions MUST match the env the model was
        # trained on -- the action space is MultiDiscrete([action_types, map_w,
        # map_h, energy_bins]), so a mismatched map size makes PPO.load reject the
        # model with an "Action spaces do not match" error.
        simulator = GameSimulator(
            model_path=model_path,
            algorithm=algorithm,
            map_width=map_width,
            map_height=map_height,
            max_steps=max_steps
        )

        # Run simulation
        print(f"\n  Running {num_episodes} episodes (this may take a while)...")
        stats_list = simulator.run_simulation(
            num_episodes=num_episodes,
            forced_opponent_types=opponents,
            render=False,
            verbose=False  # Suppress detailed output
        )

        # Analyze results - count placement finishes
        first_place = 0
        second_place = 0
        third_place = 0

        for episode_stats in stats_list:
            # Rank all participants for this episode: credits descending, then
            # nutrinium descending, then ships destroyed (kills) descending, then
            # energy ascending. This avoids awarding 1st place to PLAYER purely
            # because it was inserted first (stable-sort artifact).
            participants = []
            participants.append((
                'PLAYER',
                episode_stats.get('player_credits', 0) or 0,
                episode_stats.get('player_nutrinium', 0) or 0,
                episode_stats.get('player_kills', 0) or 0,
                episode_stats.get('player_energy', 0) or 0
            ))

            for enemy in episode_stats.get('enemy_details', []):
                e_name = enemy.get('name', 'ENEMY')
                e_credits = enemy.get('credits', 0) or 0
                e_nutrinium = enemy.get('nutrinium', 0) or 0
                e_kills = enemy.get('kills', 0) or 0
                e_energy = enemy.get('energy', 0) or 0
                participants.append((e_name, e_credits, e_nutrinium, e_kills, e_energy))

            # Sort: credits desc, nutrinium desc, kills desc, energy ascending
            participants.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))

            # Find PLAYER's placement
            for rank, (name, *rest) in enumerate(participants):
                if name == 'PLAYER':
                    if rank == 0:
                        first_place += 1
                    elif rank == 1:
                        second_place += 1
                    elif rank == 2:
                        third_place += 1
                    break

        print(f"\n Performance Results ({num_episodes} episodes):")
        print(f"  1st place: {first_place:3d} ({(first_place / num_episodes * 100:.1f) %})")
        print(f"  2nd place: {second_place:3d} ({(second_place / num_episodes * 100:.1f) %})")
        print(f"  3rd place: {third_place:3d} ({(third_place / num_episodes * 100:.1f) %})")

        # Print participant stats for detailed comparison
        print(f"\n Participant Stats:")
        _print_performance_participant_stats(stats_list, opponents)

        print(f"  {'=' * 70}")

        return {
            'first_place': first_place,
            'second_place': second_place,
            'third_place': third_place
        }

    except Exception as e:
        print(f" X Performance testing failed: {e}")
        import traceback
        traceback.print_exc()


def _save_model_attributes_to_csv(model, algorithm, model_path, callback, is_transfer,
                                  total_timesteps, version, reward_config=None, csv_filename='model_tracking.csv',
                                  performance_results=None):
    """
    Save model attributes and training metadata to CSV file for tracking.

    This function now handles schema evolution: if the existing CSV has a different
    set of headers, it will rewrite the CSV with the union of old and new headers,
    migrating existing rows and ensuring the new header order is used.
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Collect model attributes
        # Serialize reward configuration components for tracking
        try:
            import json as _json
            reward_components_serialized = _json.dumps(
                getattr(reward_config, 'composite_components', None)) if reward_config is not None else None
            reward_use_composite = bool(
                getattr(reward_config, 'use_composite', False)) if reward_config is not None else None
        except Exception:
            reward_components_serialized = None
            reward_use_composite = None

        attributes = {
            'timestamp': timestamp,
            'version': version,
            'algorithm': algorithm,
            'model_path': model_path,
            'total_timesteps': total_timesteps,
            'is_transfer_learning': is_transfer,
            'num_timesteps': getattr(model, 'num_timesteps', None),
            'reward_use_composite': reward_use_composite,
            'reward_components': reward_components_serialized,
        }

        # Get algorithm-specific hyperparameters (add in a predictable order)
        algo_fields = {}
        if algorithm == 'PPO':
            algo_fields = {
                'learning_rate': model.learning_rate if not callable(model.learning_rate) else 'schedule',
                'n_steps': model.n_steps,
                'batch_size': model.batch_size,
                'n_epochs': model.n_epochs,
                'gamma': model.gamma,
                'gae_lambda': model.gae_lambda,
                'clip_range': model.clip_range if not callable(model.clip_range) else 'schedule',
            }
        elif algorithm == 'DQN':
            algo_fields = {
                'learning_rate': model.learning_rate if not callable(model.learning_rate) else 'schedule',
                'buffer_size': model.buffer_size,
                'learning_starts': model.learning_starts,
                'batch_size': model.batch_size,
                'gamma': model.gamma,
                'train_freq': model.train_freq,
                'target_update_interval': model.target_update_interval,
                'exploration_fraction': model.exploration_fraction,
                'exploration_initial_eps': model.exploration_initial_eps,
                'exploration_final_eps': model.exploration_final_eps,
            }
        elif algorithm == 'A2C':
            algo_fields = {
                'learning_rate': model.learning_rate if not callable(model.learning_rate) else 'schedule',
                'n_steps': model.n_steps,
                'gamma': model.gamma,
                'gae_lambda': model.gae_lambda,
            }

        # Update attributes with algorithm fields in this order
        attributes.update(algo_fields)

        # Add training performance metrics
        if hasattr(callback, 'episode_rewards') and callback.episode_rewards:
            attributes.update({
                'total_episodes': len(callback.episode_rewards),
                'avg_reward': float(np.mean(callback.episode_rewards)),
                'std_reward': float(np.std(callback.episode_rewards)),
                'max_reward': float(np.max(callback.episode_rewards)),
                'min_reward': float(np.min(callback.episode_rewards)),
                'final_10_avg_reward': float(np.mean(callback.episode_rewards[-10:])) if len(
                    callback.episode_rewards) >= 10 else float(np.mean(callback.episode_rewards)),
            })
        else:
            attributes.update({
                'total_episodes': 0,
                'avg_reward': 0.0,
                'std_reward': 0.0,
                'max_reward': 0.0,
                'min_reward': 0.0,
                'final_10_avg_reward': 0.0,
            })

        # Add performance test results (simulation against existing models)
        if performance_results:
            attributes.update({
                'perf_1st_place': performance_results.get('first_place', 0),
                'perf_2nd_place': performance_results.get('second_place', 0),
                'perf_3rd_place': performance_results.get('third_place', 0),
            })
        else:
            attributes.update({
                'perf_1st_place': 0,
                'perf_2nd_place': 0,
                'perf_3rd_place': 0,
            })

        # Determine CSV handling: migrate if schema changed
        file_exists = os.path.isfile(csv_filename)

        if not file_exists:
            # Write new file with header and row
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(attributes.keys()))
                writer.writeheader()
                writer.writerow(attributes)
            print(f"Model attributes saved to {csv_filename}")
            return

        # If file exists, read existing header
        with open(csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

        new_fieldnames = list(attributes.keys())

        # If existing header differs (different set or order), rewrite file with union
        if set(existing_fieldnames) != set(new_fieldnames):
            # Create union preserving new_fieldnames order, then append any existing-only fields
            union = list(new_fieldnames)
            for fn in existing_fieldnames:
                if fn not in union:
                    union.append(fn)

            # Migrate existing rows into new format (fill missing with empty strings)
            migrated_rows = []
            for row in existing_rows:
                new_row = {fn: row.get(fn, '') for fn in union}
                migrated_rows.append(new_row)

            # Append the new attributes row
            new_row = {fn: attributes.get(fn, '') for fn in union}
            migrated_rows.append(new_row)

            # Write back the file with updated header
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=union)
                writer.writeheader()
                for r in migrated_rows:
                    writer.writerow(r)

            print(f"Model attributes saved to {csv_filename} (schema migrated, header updated)")
            return

        # If headers match (same set), append using existing order
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=existing_fieldnames)
            # Ensure we only write keys that exist in existing_fieldnames (ignore extras)
            row = {fn: attributes.get(fn, '') for fn in existing_fieldnames}
            writer.writerow(row)

        print(f"Model attributes appended to {csv_filename}")

    except Exception as e:
        print(f"Warning: Could not save model attributes to CSV: {e}")
        import traceback
        traceback.print_exc()


def train_with_sb3(algorithm='PPO', total_timesteps=100000, save_path='models/',
                   transfer_from=None, freeze_layers=False, fine_tune_lr=None,
                   map_width=10, map_height=10, min_opponents=2, max_opponents=2, max_steps=300,
                   use_predefined_asteroids=False, asteroid_config_path='asteroids.config',
                   use_predefined_start=False, start_position_config_path='start_positions.config',
                   use_composite=True, composite_components: Optional[List[object]] = None,
                   efficiency_mode=False, num_threads=None):
    """
    Train agent using Stable Baselines3 with optional transfer learning

    Args:
        algorithm: RL algorithm to use ('PPO', 'DQN', 'A2C')
        total_timesteps: Total timesteps for training
        save_path: Directory to save models
        transfer_from: Path to pre-trained model for transfer learning (optional)
        freeze_layers: If True, freeze early layers during transfer learning
        fine_tune_lr: Custom learning rate for fine-tuning (default: lower than normal)
        map_width: Width of the game map
        map_height: Height of the game map
        min_opponents: Minimum number of opponent ships
        max_opponents: Maximum number of opponent ships
        max_steps: Maximum steps per episode
        use_predefined_asteroids: Use predefined asteroids from config file
        asteroid_config_path: Path to asteroid configuration file
        use_predefined_start: Use predefined starting positions from config file
        start_position_config_path: Path to starting position configuration file
        use_composite: Use composite reward calculator
        composite_components: List of reward components for composite calculator
        efficiency_mode: If True, limit CPU usage for efficiency. If False, maximize CPU usage for faster training
        num_threads: Specific number of CPU threads to use (overrides efficiency_mode if set)
        """

    if not SB3_AVAILABLE:
        print("Please install stable-baselines3 first:")
        print("pip install stable-baselines3[extra]")
        return None

    is_transfer = transfer_from is not None

    print("=" * 70)
    if is_transfer:
        print(f"TRANSFER LEARNING: {algorithm} AGENT WITH STABLE BASELINES3")
        print(f"Loading pre-trained model from: {transfer_from}")
    else:
        print(f"TRAINING {algorithm} AGENT WITH STABLE BASELINES3")
    print("=" * 70)

    # Set process title for Task Manager visibility
    if SETPROCTITLE_AVAILABLE:
        setproctitle("PnP Training")
    # Set console window title (more visible on Windows)
    set_console_title("PnP Training")

    # Configure CPU usage mode
    cpu_settings = set_cpu_mode(efficiency_mode=efficiency_mode, num_threads=num_threads)
    print(f"\nCPU Configuration:")
    print(f"  Mode: {cpu_settings['mode']}")
    print(f"  Threads: {cpu_settings['threads']}/{cpu_settings['cpu_count']} CPUs")
    if cpu_settings.get('torch_configured'):
        print(f" PyTorch: Configured for {cpu_settings['threads']} threads")
    if cpu_settings.get('priority_set'):
        print(f" Process Priority: {cpu_settings.get('process_priority', 'default')}")
    if efficiency_mode:
        print(f"  Efficiency Mode: Balanced CPU usage for background training")
    else:
        print(f"  Performance Mode: Maximum CPU usage for fastest training")

    # Create environment
    # Build reward config to pass into the environment
    reward_cfg = RewardConfig()
    reward_cfg.use_composite = bool(use_composite)
    reward_cfg.composite_components = composite_components

    # Determine opponent handling strategy
    use_dynamic_opponents = (min_opponents != max_opponents)

    if use_dynamic_opponents:
        # Start with mid-range opponent count for initial env creation
        initial_num_opponents = (min_opponents + max_opponents) // 2
        print(f"\nOpponent Configuration:")
        print(f"  Dynamic per-episode sampling: {min_opponents} to {max_opponents} opponents")
        print(f"  Initial env uses: {initial_num_opponents} opponents")
    else:
        # Fixed opponent count
        initial_num_opponents = min_opponents
        print(f"\nOpponent Configuration:")
        print(f"  Fixed count: {initial_num_opponents} opponents per episode")

    env = ProspectorsPiratesEnv(
        map_width=map_width,
        map_height=map_height,
        num_opponents=initial_num_opponents,
        max_steps=max_steps,
        use_predefined_asteroids=use_predefined_asteroids,
        asteroid_config_path=asteroid_config_path,
        use_predefined_start=use_predefined_start,
        start_position_config_path=start_position_config_path,
        reward_config=reward_cfg
    )

    # Apply dynamic opponents wrapper if using variable opponent counts
    if use_dynamic_opponents:
        env = DynamicOpponentsWrapper(env, min_opponents, max_opponents)

    # Apply action mask wrapper for standard PPO.
    # This intercepts invalid actions, replaces them with a random valid action,
    # and applies a penalty so the model learns to respect the mask.
    env = ActionMaskWrapper(env)
    print("  Action masking: penalty-based enforcement")

    # Wrap environment with Monitor for logging
    env = Monitor(env)

    # Check environment
    print("\nChecking environment compatibility...")
    try:
        check_env(env, warn=True)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return None

    os.makedirs(save_path, exist_ok=True)

    # Initialize model variable
    model = None
    original_env = env  # Keep reference in case transfer loading modifies env

    # Transfer Learning: Load existing model
    if is_transfer:
        print(f"\nLoading pre-trained {algorithm} model for transfer learning...")
        print(f"   Model path: {transfer_from}")

        # Check if model file exists
        model_path_with_zip = transfer_from.endswith('.zip') else f"{transfer_from}.zip"
        if not os.path.exists(model_path_with_zip):
            print(f"x Model file not found: {model_path_with_zip}")
            print("   Available models:")
            if os.path.exists('../models'):
                for f in os.listdir('../models'):
                    if f.endswith('.zip'):
                        print(f" - models/{f.replace('.zip', '')}")
            print("   Falling back to training from scratch...")
            is_transfer = False
        else:
            try:
                print(f"  Loading from: {model_path_with_zip}")

                if algorithm == 'PPO':
                    # Load as regular PPO.
                    # The env uses a Dict obs space ({'observation', 'action_mask'}).
                    # Regular PPO with MultiInputPolicy can handle Dict observations.
                    # Action masking is enforced by the ActionMaskWrapper.
                    model = PPO.load(transfer_from, env=env)
                    print("   ✓ Loaded as regular PPO")
                elif algorithm == 'DQN':
                    model = DQN.load(transfer_from, env=env)
                elif algorithm == 'A2C':
                    model = A2C.load(transfer_from, env=env)
                else:
                    print(f"x  Unknown algorithm: {algorithm}")
                    return None

                model.verbose = 0  # Disable verbose output during evaluation

                # Fix clip_range schedule corruption:
                # When SB3 saves a model whose clip_range is a schedule (lambda),
                # loading it can double-wrap the schedule, producing a nested lambda
                # that crashes with "float expected at most 1 argument, got 2".
                # Conversely, if the schedule is unwrapped to a bare float, PPO.train()
                # crashes with "'float' object is not callable".
                # Fix: evaluate the current schedule to get the float value, then
                # re-wrap via get_schedule_fn so it's a proper single-layer callable.
                if algorithm in ('PPO',):
                    from stable_baselines3.common.utils import get_schedule_fn
                    if hasattr(model, 'clip_range'):
                        try:
                            val = float(model.clip_range(1.0)) if callable(model.clip_range) else float(
                                model.clip_range)
                        except Exception as e:
                            val = 0.2  # PPO default
                        model.clip_range = get_schedule_fn(val)
                    if hasattr(model, 'clip_range_vf') and model.clip_range_vf is not None:
                        try:
                            val = float(model.clip_range_vf(1.0)) if callable(model.clip_range_vf) else float(
                                model.clip_range_vf)
                        except Exception as e:
                            val = None
                        model.clip_range_vf = get_schedule_fn(val) if val is not None else None

                print(f"✓ Successfully loaded model from {transfer_from}")
                print(f"   Model has {model.num_timesteps} training timesteps")

                # Set fine-tuning learning rate for transfer learning.
                # IMPORTANT: Always use a fixed (non-schedule) learning rate.
                # Inheriting a decaying schedule from the saved model causes the lr
                # to collapse toward zero after repeated transfers, producing
                # "dead" models that can no longer learn.
                _default_finetune_lr = {'PPO': 3e-5, 'DQN': 1e-5, 'A2C': 7e-5}
                _min_lr = 1e-6  # Floor: never go below this

                if fine_tune_lr is not None:
                    # User explicitly provided a learning rate - use it as-is
                    model.learning_rate = fine_tune_lr
                    print(f"   Using custom learning rate: {fine_tune_lr}")
                else:
                    original_lr = model.learning_rate
                    default_ft = _default_finetune_lr.get(algorithm, 3e-5)

                    if callable(original_lr):
                        # Learning rate is a schedule function from the saved model.
                        # Replace with a fixed rate to prevent decay across transfers.
                        model.learning_rate = default_ft
                        print(f"   Replaced learning rate schedule with fixed rate: {default_ft:.2e}")
                    else:
                        # Use the larger of: stored_lr / 10, or the default fine-tune rate
                        # (both floored at _min_lr). If the stored LR was already very
                        # small (from repeated transfers), fall back to the algorithm's
                        # standard fine-tuning rate.
                        candidate = max(original_lr / 10, _min_lr)
                        if candidate < default_ft:
                            model.learning_rate = default_ft
                            print(
                                f"   Learning rate was too small ({original_lr:.2e}), using default fine-tuning rate: {default_ft:.2e}")
                        else:
                            model.learning_rate = candidate
                            print(
                                f" Reduced learning rate for fine-tuning: {original_lr:.2e} -> {model.learning_rate:.2e}"
                            )

                # Optionally freeze early layers
                if freeze_layers:
                    print(" Freezing early network layers...")
                    _freeze_early_layers(model, algorithm)
                    print(" ✓ Early layers frozen")

            except Exception as e:
                print(f"x Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                print(" Falling back to training from scratch...")
                is_transfer = False
                model = None
                env = original_env  # Restore original env for fresh model creation

    # Create new model if not transfer learning
    if not is_transfer:
        print(f"\nCreating new {algorithm} model...")

        if algorithm == 'PPO':
            model = PPO(
                'MultiInputPolicy',
                env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=2048,  # 2048 steps ≈ 6-7 episodes per rollout (optimized for 300-step episodes)
                batch_size=256,  # 256 samples per gradient update for stable learning
                n_epochs=10,  # Increased to 10 for better sample efficiency (standard PPO)
                gamma=0.99,  # Standard discount factor (0.99 works well for 300-step horizon)
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # Entropy bonus to encourage exploration
                vf_coef=0.5,  # Value function coefficient
                max_grad_norm=0.5,  # Gradient clipping for stability
                policy_kwargs=dict(
                    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                    # Larger network for expanded sensor range (11x11 grid)
                ),
                tensorboard_log="./tensorboard_logs/ppo/"
            )
        elif algorithm == 'DQN':
            model = DQN(
                'MultiInputPolicy',
                env,
                verbose=1,
                learning_rate=1e-4,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                tensorboard_log="./tensorboard_logs/dqn/"
            )
        elif algorithm == 'A2C':
            model = A2C(
                'MultiInputPolicy',
                env,
                verbose=1,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                tensorboard_log="./tensorboard_logs/a2c/"
            )
        else:
            print(f"Unknown algorithm: {algorithm}")
            return None

    # Create callback
    callback = TrainingCallback(verbose=1, total_timesteps=total_timesteps)

    # Add checkpoint callback if training for more than 1M timesteps
    callbacks = [callback]
    if total_timesteps > 1_000_000:
        checkpoint_callback = CheckpointCallback(
            save_freq=1_000_000,  # Save every 1 million timesteps
            save_path=save_path,
            algorithm=algorithm,
            name_prefix='checkpoint',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
    print(f"\n✓ Checkpoint saving enabled: every 1M timesteps")
    print(f" Expected checkpoints: {total_timesteps // 1_000_000}")

    # Ensure model was created successfully
    if model is None:
        print("x Failed to create or load model. Aborting training.")
        return None

    # Train model
    training_type = "fine-tuning" if is_transfer else "training"
    print(f"\n{training_type.capitalize()} for {total_timesteps:,} timesteps...")
    try:
        # Always reset timestep counter so that total_timesteps means
        # "train for N additional steps" (not "train until N total").
        # Without this, transfer-learning from a 30M-step model with
        # --timesteps 100000 would immediately finish (30M > 100K).
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True
        )
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current model...")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nAttempting to save current model state...")
        # Continue to save even if training failed

    # Save model
    # Get version number
    version = _get_next_version_number(algorithm, save_path, is_transfer)
    model_path = os.path.join(save_path, f'{algorithm.lower()}_{pn_model_v[version]}')

    # Save model FIRST so it can be loaded for performance testing
    _safe_model_save(model, model_path)
    print(f"\nModel saved to {model_path} (version {version})")

    # Run performance test against existing models. Pass the SAME map dimensions
    # used for training so the simulation env's action space matches the model.
    performance_results = _test_model_performance(
        model_path=model_path,
        algorithm=algorithm,
        num_episodes=10,
        map_width=map_width,
        map_height=map_height,
        max_steps=max_steps
    )

    # Save model attributes to CSV for tracking (including performance results)
    _save_model_attributes_to_csv(
        model=model,
        algorithm=algorithm,
        model_path=model_path,
        callback=callback,
        is_transfer=is_transfer,
        total_timesteps=total_timesteps,
        version=version,
        reward_config=reward_cfg,
        csv_filename='../model_tracking.csv',
        performance_results=performance_results
    )

    # Plot training progress
    if callback.episode_credits:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(callback.episode_credits)
        plt.title('Episode Credits')
        plt.xlabel('Episode')
        plt.ylabel('Credits Earned')
        plt.grid(True)

        # Moving average
        window = 10
        if len(callback.episode_credits) > window:
            moving_avg = np.convolve(callback.episode_credits,
                                     np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(callback.episode_credits))),
            moving_avg, 'r-', linewidth=2, label=f'{window}-episode MA')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(callback.episode_credits, bins=20, edgecolor='black')
        plt.title('Credits Distribution')
        plt.xlabel('Credits')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_filename = f'output/training_progress/{algorithm.lower()}_{version}_training_progress.png'
        plt.savefig(plot_filename)
        print(f"Training plots saved to {plot_filename}")
        plt.close()

    env.close()
    return model

def _parse_model_path_with_spec(raw_model_path: str):
    """Parse model path token supporting optional MODEL_PATH::SPEC_NAME syntax."""
    token = (raw_model_path or '').strip()
    if not token:
        return token, None
    if '::' not in token:
        return token, None
    model_path, spec_name = token.split('::', 1)
    return model_path.strip(), (spec_name.strip() or None)


def evaluate_model(model, num_episodes=10, render=False, min_opponents=2, max_opponents=2, player_model_spec=None):
    """Evaluate trained model"""

    if not SB3_AVAILABLE:
        return

    print("\n" + "=" * 70)
    print("EVALUATING TRAINED MODEL")
    print("=" * 70)

    # Set process title for Task Manager visibility
    if SETPROCTITLE_AVAILABLE:
        setproctitle("PnP Playing")
    # Set console window title (more visible on Windows)
    set_console_title("PnP Playing")

    # Determine opponent handling strategy
    use_dynamic_opponents = (min_opponents != max_opponents)

    if use_dynamic_opponents:
        initial_num_opponents = (min_opponents + max_opponents) // 2
        print(f"Opponent count: {min_opponents}-{max_opponents} (dynamic per episode)")
    else:
        initial_num_opponents = min_opponents
        print(f"Opponent count: {initial_num_opponents} (fixed)")

    env = ProspectorsPiratesEnv(
        map_width=10,
        map_height=10,
        num_opponents=initial_num_opponents,
        max_steps=300,
        render_mode='human' if render else None
    )

    if player_model_spec is not None:
        env.player_model_spec = player_model_spec

    # Apply dynamic opponents wrapper if using variable opponent counts
    if use_dynamic_opponents:
        env = DynamicOpponentsWrapper(env, min_opponents, max_opponents)

    # If the model expects a flat Box observation, wrap the env so observations match.
    model_obs_space = getattr(model, 'observation_space', None)
    if model_obs_space is not None and not isinstance(model_obs_space, gym.spaces.Dict):
        env = FlattenDictObsWrapper(env)

    episode_rewards = []
    episode_credits = []
    episode_nutrinium = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if render and step % 20 == 0:
                env.render()
            step += 1

        if render:
            env.render()

        episode_rewards.append(episode_reward)
        episode_credits.append(info['player_credits'])
        episode_nutrinium.append(info['player_nutrinium'])

        print(f"\nEpisode {episode + 1}/{num_episodes}:")
        print(f"   Total Reward: {episode_reward:.2f}")
        print(f"   Credits Earned: {info['player_credits']}")
        print(f"   Final Nutrinium: {info['player_nutrinium']}")
        print(f"   Steps: {step}")
        print(f"   Destroyed: {info['player_destroyed']}")

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Credits: {np.mean(episode_credits):.1f} ± {np.std(episode_credits):.1f}")
    print(f"Average Nutrinium: {np.mean(episode_nutrinium):.1f} ± {np.std(episode_nutrinium):.1f}")
    print(f"Max Credits: {np.max(episode_credits)}")
    print("=" * 70)

    env.close()


def compare_algorithms(timesteps_per_algorithm=50000, min_opponents=2, max_opponents=2):
    """Compare different algorithms"""

    if not SB3_AVAILABLE:
        return

    print("\n" + "=" * 70)
    print("COMPARING DIFFERENT RL ALGORITHMS")
    print("=" * 70)

    algorithms = ['PPO', 'DQN', 'A2C']
    results = {}

    for algo in algorithms:
        print(f"\n{'=' * 70}")
        print(f"Training {algo}...")
        print(f"{'=' * 70}")

        model = train_with_sb3(algo, total_timesteps=timesteps_per_algorithm)

        if model:
            # Evaluate
            use_dynamic_opponents = (min_opponents != max_opponents)
            initial_num_opponents = (min_opponents + max_opponents) // 2 if use_dynamic_opponents else min_opponents

            env = ProspectorsPiratesEnv(
                map_width=10,
                map_height=10,
                num_opponents=initial_num_opponents,
                max_steps=300,
            )

            # Apply dynamic opponents wrapper if using variable opponent counts
            if use_dynamic_opponents:
                env = DynamicOpponentsWrapper(env, min_opponents, max_opponents)

            # Flatten obs if the model expects a flat Box (MlpPolicy)
            model_obs_space = getattr(model, 'observation_space', None)
            if model_obs_space is not None and not isinstance(model_obs_space, gym.spaces.Dict):
                env = FlattenDictObsWrapper(env)
            env = Monitor(env)

            episode_rewards = []
            episode_credits = []

            for _ in range(20):
                obs, _ = env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward

                episode_rewards.append(episode_reward)
                episode_credits.append(info['player_credits'])

            results[algo] = {
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'avg_credits': np.mean(episode_credits),
                'std_credits': np.std(episode_credits),
            }

            env.close()

    # Print comparison
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 70)
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        print(f"  Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Average Credits: {metrics['avg_credits']:.1f} ± {metrics['std_credits']:.1f}")
    print("=" * 70)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    algos = list(results.keys())
    rewards = [results[a]['avg_reward'] for a in algos]
    reward_stds = [results[a]['std_reward'] for a in algos]
    credits = [results[a]['avg_credits'] for a in algos]
    credit_stds = [results[a]['std_credits'] for a in algos]

    ax1.bar(algos, rewards, yerr=reward_stds, capsize=5)
    ax1.set_title('Average Reward by Algorithm')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True, alpha=0.3)

    ax2.bar(algos, credits, yerr=credit_stds, capsize=5)
    ax2.set_title('Average Credits by Algorithm')
    ax2.set_ylabel('Average Credits')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    print("\nComparison plot saved to algorithm_comparison.png")
    plt.close()


if __name__ == "__main__":
    if not SB3_AVAILABLE:
        print("\nPlease install Stable Baselines3:")
        print("pip install stable-baselines3[extra]")
        exit(1)

    # Set process title early for Task Manager visibility
    if SETPROCTITLE_AVAILABLE:
        setproctitle("PnP Training")
    # Set console window title (more visible on Windows)
    set_console_title("PnP Training")

    import argparse

    parser = argparse.ArgumentParser(
        description='Train RL agent for Prospectors n Pirates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""""
Examples:
    # Train a new PPO model
    python example_sb3.py --algorithm PPO --timesteps 100000
    
    # Long training with automatic checkpoints (saves every 1M timesteps)
    python example_sb3.py --algorithm PPO --timesteps 50000000
    
    # Train with efficiency mode (lower CPU usage, good for background training)
    python example_sb3.py --algorithm PPO --timesteps 100000 --efficiency-mode
    
    # Train with maximum CPU usage (default - fastest training)
    python example_sb3.py --algorithm PPO --timesteps 100000
    
    # Train with custom thread count
    python example_sb3.py --algorithm PPO --timesteps 100000 --num-threads 4
    
    # Train with custom map size and opponents
    python example_sb3.py --algorithm PPO --timesteps 100000 --map-width 15 --map-height 15 --min-opponents 1 --max-opponents 5
    
    # Train with predefined asteroids
    python example_sb3.py --algorithm PPO --timesteps 100000 --predefined-asteroids --asteroid-config asteroids.config
    
    # Transfer learning: fine-tune an existing model
    python example_sb3.py --algorithm PPO --timesteps 50000 --transfer-from models/ppo_pnp_model
    
    # Transfer learning with frozen layers and custom learning rate
    python example_sb3.py --algorithm PPO --timesteps 50000 --transfer-from models/ppo_pnp_model --freeze-layers --learning-rate 0.00001
    
    # Train on larger map with more opponents and predefined asteroids
    python example_sb3.py --algorithm PPO --timesteps 200000 --map-width 20 --map-height 20 --min-opponents 3 --max-opponents 5 --max-steps 500 --predefined-asteroids
    
    # Evaluate a trained model
    python example_sb3.py --algorithm PPO --evaluate --model-path models/ppo_pnp_model --render
"""
    )
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'DQN', 'A2C', 'compare'],
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps for training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate a trained model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model for evaluation. Optional syntax: MODEL_PATH::SPEC_NAME')
    parser.add_argument('--render', action='store_true',
                        help='Render during evaluation')

    # Transfer learning arguments
    parser.add_argument('--transfer-from', type=str, default=None,
                        help='Path to pre-trained model for transfer learning')
    parser.add_argument('--freeze-layers', action='store_true',
                        help='Freeze early network layers during transfer learning')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Custom learning rate for fine-tuning (default: auto-reduced)')

    # Reward composite options
    parser.add_argument('--no-composite', action='store_true',
                        help='Disable composite reward calculator (use simple RewardCalculator)')
    parser.add_argument('--reward-components', type=str, default=None,
                        help='JSON string specifying reward components (list of names or dicts)')
    parser.add_argument('--reward-components-file', type=str, default=None,
                        help='Path to JSON file specifying reward components')

    # Environment configuration arguments
    parser.add_argument('--map-width', type=int, default=10,
                        help='Width of the game map (default: 10)')
    parser.add_argument('--map-height', type=int, default=10,
                        help='Height of the game map (default: 10)')
    parser.add_argument('--min-opponents', type=int, default=2,
                        help='Minimum number of opponent ships (default: 2)')
    parser.add_argument('--max-opponents', type=int, default=2,
                        help='Maximum number of opponent ships (default: 2)')
    parser.add_argument('--max-steps', type=int, default=300,
                        help='Maximum steps per episode (default: 300)')
    parser.add_argument('--predefined-asteroids', action='store_true',
                        help='Use predefined asteroids from config file instead of random generation')
    parser.add_argument('--asteroid-config', type=str, default='asteroids_with_trading_posts.config',
                        help='Path to asteroid configuration file (default: asteroids_with_trading_posts.config)')
    parser.add_argument('--predefined-start', action='store_true',
                        help='Use predefined starting positions from config file')
    parser.add_argument('--start-position-config', type=str, default='start_positions.config',
                        help='Path to starting position configuration file (default: start_positions.config)')
    # CPU/Performance options
    parser.add_argument('--efficiency-mode', action='store_true',
                        help='Run in efficiency mode (lower CPU usage, slower training) - good for background training')
    parser.add_argument('--num-threads', type=int, default=None,
                        help='Specific number of CPU threads to use (overrides efficiency mode)')
    args = parser.parse_args()

    if args.evaluate:
        if args.model_path:
            eval_model_path, eval_spec_name = _parse_model_path_with_spec(args.model_path)
            player_model_spec = DEFAULT_FULL_SPEC
            if eval_spec_name:
                resolved_spec = get_named_model_spec(eval_spec_name)
                if resolved_spec is not None:
                    player_model_spec = resolved_spec
                else:
                    print(f"WARNING: Unknown player MODEL_SPEC '{eval_spec_name}' in --model-path. Falling back to DEFAULT_FULL_SPEC.")
            if args.algorithm == 'PPO':
                model = PPO.load(eval_model_path)
                print("Loaded model as regular PPO")
            elif args.algorithm == 'DQN':
                model = DQN.load(eval_model_path)
            elif args.algorithm == 'A2C':
                model = A2C.load(eval_model_path)
            else:
                model = None
            if model is None:
                print(f"Could not load model for algorithm {args.algorithm} from {eval_model_path}")
            else:
                model.verbose = 0  # Disable verbose output during evaluation
                evaluate_model(model, num_episodes=10, render=args.render, min_opponents=args.min_opponents,
                               max_opponents=args.max_opponents, player_model_spec=player_model_spec)
        else:
            print("Please specify --model-path for evaluation")
    elif args.algorithm == 'compare':
        compare_algorithms(timesteps_per_algorithm=args.timesteps, min_opponents=args.min_opponents,
                           max_opponents=args.max_opponents)
    else:
        # Prepare reward composite options
        import json

        use_composite_flag = not args.no_composite
        composite_specs = None
        # Priority: file over inline JSON
        if args.reward_components_file:
            try:
                with open(args.reward_components_file, 'r', encoding='utf-8') as f:
                    composite_specs = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read reward components file {args.reward_components_file}: {e}")
                composite_specs = None
        elif args.reward_components:
            try:
                composite_specs = json.loads(args.reward_components)
            except Exception as e:
                print(f"Warning: Could not parse --reward-components JSON: {e}")
                composite_specs = None

        model = train_with_sb3(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            transfer_from=args.transfer_from,
            freeze_layers=args.freeze_layers,
            fine_tune_lr=args.learning_rate,
            map_width=args.map_width,
            map_height=args.map_height,
            min_opponents=args.min_opponents,
            max_opponents=args.max_opponents,
            max_steps=args.max_steps,
            use_predefined_asteroids=args.predefined_asteroids,
            asteroid_config_path=args.asteroid_config,
            use_predefined_start=args.predefined_start,
            start_position_config_path=args.start_position_config,
            use_composite=use_composite_flag,
            composite_components=composite_specs,
            efficiency_mode=args.efficiency_mode,
            num_threads=args.num_threads
        )
        if model:
            # After training and saving the model, ask the user whether to proceed to evaluation
            cont = _ask_to_evaluate()
            if cont:
                evaluate_model(model, num_episodes=5, render=True, min_opponents=args.min_opponents,
                               max_opponents=args.max_opponents)
