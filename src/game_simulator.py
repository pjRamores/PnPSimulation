"""
Game Simulator for Prospectors n Pirates

This module provides a class for simulating games using trained RL models
against varying numbers of opponents.
"""

import numpy as np
import random
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple

try:
    from stable_baselines3 import PPO, DQN, A2C
    SB3_AVAILABLE = True
except ImportError:
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

from pnp_env import ProspectorsPiratesEnv, OpponentAIType
from model_adapter import wrap_model_with_compat
from model_specs import DEFAULT_FULL_SPEC, get_named_model_spec


def _parse_model_path_with_spec(raw_model_path: str) -> Tuple[str, Optional[str]]:
    """Parse model path token supporting optional MODEL_PATH::SPEC_NAME syntax."""
    token = (raw_model_path or '').strip()
    if not token:
        return token, None
    if '::' not in token:
        return token, None
    model_path, spec_name = token.split('::', 1)
    return model_path.strip(), (spec_name.strip() or None)


def _expand_opponents_with_counts(raw_opponents: str) -> List[str]:
    """Expand a comma-separated --opponents string into a flat opponent list.

    Each comma-separated token may be a plain opponent name / model path (which
    contributes a single entry) or carry a trailing ``[N]`` repeat-count suffix
    that repeats it N times. For example ``BOT_V2[1],BOT_V3[3],BOT_V4,BOT_V5[3]``
    expands to one BOT_V2, three BOT_V3, one BOT_V4 and three BOT_V5 (equivalent
    to listing each name explicitly). Whitespace around tokens and inside the
    brackets is ignored. A count of 0 omits the opponent; a malformed bracket
    raises ValueError.
    """
    import re

    # name = everything up to an optional "[N]" suffix; name may contain "::SPEC".
    pattern = re.compile(r'^(?P<name>.*?)\s*\[\s*(?P<count>\d+)\s*\]$')
    expanded: List[str] = []
    for token in raw_opponents.split(','):
        token = token.strip()
        if not token:
            continue
        match = pattern.match(token)
        if match:
            name = match.group('name').strip()
            if not name:
                raise ValueError(f"Invalid opponent token (missing name): '{token}'")
            count = int(match.group('count'))
            expanded.extend([name] * count)
        elif '[' in token or ']' in token:
            # Bracket present but not a well-formed "[N]" suffix.
            raise ValueError(
                f"Invalid opponent count syntax: '{token}'. Expected NAME[N], e.g. BOT_V3[3]."
            )
        else:
            expanded.append(token)
    return expanded


class SimulationLogger:
    """Logger that writes to both console and file(s)."""

    def __init__(self, output_dir: str, enable_logging: bool = True):
        """
        Initialize the logger.

        Args:
            output_dir: Directory where log files will be saved
            enable_logging: Whether to enable file logging
        """
        self.enable_logging = enable_logging
        self.output_dir = output_dir
        self.simulation_log_path = None
        self.current_episode_log = None
        self.current_episode_number = None

        if self.enable_logging:
            os.makedirs(output_dir, exist_ok=True)
            self.simulation_log_path = os.path.join(output_dir, "simulation_log.txt")
            # Create/clear the simulation log file
            with open(self.simulation_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Simulation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")

    def start_episode(self, episode_number: int, total_episodes: int):
        """Start logging a new episode."""
        if not self.enable_logging:
            return

        self.current_episode_number = episode_number
        # Determine padding based on total episodes
        padding = len(str(total_episodes))
        episode_filename = f"episode_{str(episode_number).zfill(padding)}.txt"

        # If more than 100 episodes, organize into subfolders
        if total_episodes > 100:
            # Calculate which subfolder this episode belongs to (1-100, 101-200, etc.)
            folder_start = ((episode_number - 1) // 100) * 100 + 1
            folder_end = min(folder_start + 99, total_episodes)
            subfolder_name = f"episodes_{str(folder_start).zfill(padding)}-{str(folder_end).zfill(padding)}"
            episode_dir = os.path.join(self.output_dir, subfolder_name)
            os.makedirs(episode_dir, exist_ok=True)
            self.current_episode_log = os.path.join(episode_dir, episode_filename)
        else:
            # No subfolder needed for <= 100 episodes
            self.current_episode_log = os.path.join(self.output_dir, episode_filename)

        # Create/clear the episode log file
        with open(self.current_episode_log, 'w', encoding='utf-8') as f:
            f.write(f"Episode {episode_number} Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

    def end_episode(self):
        """End the current episode logging."""
        self.current_episode_log = None
        self.current_episode_number = None

    def log(self, message: str, to_episode: bool = True, to_simulation: bool = True, to_console: bool = True):
        """
        Write a message to console and/or log file(s).

        Args:
            message: Message to log
            to_episode: Whether to write to current episode log
            to_simulation: Whether to write to simulation log
            to_console: Whether to print to console (NEW)
        """
        # Print to console only if requested
        if to_console:
            print(message)

        if not self.enable_logging:
            return

        # Write to simulation log
        if to_simulation and self.simulation_log_path:
            try:
                with open(self.simulation_log_path, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except Exception as e:
                print(f"Warning: Failed to write to simulation log: {e}")

        # Write to episode log
        if to_episode and self.current_episode_log:
            try:
                with open(self.current_episode_log, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except Exception as e:
                print(f"Warning: Failed to write to episode log: {e}")

    def close(self):
        """Close the logger."""
        if self.enable_logging and self.simulation_log_path:
            with open(self.simulation_log_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"Simulation completed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


class GameSimulator:
    """Simulator for running trained models against opponents"""

    # Action names for display
    ACTION_NAMES = [
        "WAIT", "MINE", "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
        "RECHARGE", "RECHARGE_END", "ATTACK", "JUMP_TO_ASTEROID", "SELL",
        "RAISE_SHIELDS", "JUMP_TO_TRADING_POST", "RESPAWN",
        "PLUNDER", "SALVAGE", "REPAIR", "NEGOTIATE", "LOWER_SHIELDS"
    ]

    def __init__(self,
                 model_path: str,
                 algorithm: str = "PPO",
                 map_width: int = 10,
                 map_height: int = 10,
                 max_steps: int = 300,
                 use_predefined_asteroids: bool = False,
                 asteroid_config_path: str = 'asteroids.config',
                 use_predefined_start: bool = False,
                 start_position_config_path: str = 'start_positions.config',
                 enable_logging: bool = True,
                 output_base_dir: str = 'output'):
        """
        Initialize the game simulator.

        Args:
            model_path: Path to the trained model file
            algorithm: Algorithm used (PPO, DQN, A2C)
            map_width: Width of the game map
            map_height: Height of the game map
            max_steps: Maximum steps per episode
            use_predefined_asteroids: Use predefined asteroids from config file
            asteroid_config_path: Path to asteroid configuration file
            use_predefined_start: Use predefined starting positions from config file
            start_position_config_path: Path to starting position configuration file
            enable_logging: Whether to enable logging to files
            output_base_dir: Base directory for output logs (default: 'output')
        """
        parsed_model_path, parsed_spec_name = _parse_model_path_with_spec(model_path)
        self.model_path = parsed_model_path
        self.player_model_spec = DEFAULT_FULL_SPEC
        if parsed_spec_name:
            resolved = get_named_model_spec(parsed_spec_name)
            if resolved is not None:
                self.player_model_spec = resolved
            else:
                print(f"WARNING: Unknown player MODEL_SPEC '{parsed_spec_name}' in --model-path. Falling back to DEFAULT_FULL_SPEC.")
        self.algorithm = algorithm.upper()
        self.map_width = map_width
        self.map_height = map_height
        self.max_steps = max_steps
        self.use_predefined_asteroids = use_predefined_asteroids
        self.asteroid_config_path = asteroid_config_path
        self.use_predefined_start = use_predefined_start
        self.start_position_config_path = start_position_config_path

        # Track cumulative player placements across episodes
        self.player_placements = {1: 0, 2: 0, 3: 0}  # 1st, 2nd, 3rd place counts

        # Logging setup
        self.enable_logging = enable_logging
        self.logger = None
        if enable_logging:
            # Extract model name from path
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            # Create timestamped directory name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(output_base_dir, f"{model_name}_{timestamp}")
            self.logger = SimulationLogger(output_dir, enable_logging=True)

        # Validate model exists
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")

        if not os.path.exists(self.model_path + ".zip") and not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Statistics tracking
        self.episode_stats: List[Dict] = []

    # ================================================
    # Public API
    # ================================================

    @staticmethod
    def get_ai_type_name(ai_type):
        """Convert AI type enum to readable string"""
        if ai_type == OpponentAIType.MODEL:
            return "MODEL"
        elif ai_type == OpponentAIType.BOT_V2:
            return "BOT_V2"
        elif ai_type == OpponentAIType.BOT_V3:
            return "BOT_V3"
        elif ai_type == OpponentAIType.BOT_V4:
            return "BOT_V4"
        elif ai_type == OpponentAIType.BOT_V5:
            return "BOT_V5"
        elif ai_type == OpponentAIType.BOT_V6:
            return "BOT_V6"
        elif ai_type == OpponentAIType.BOT_V7:
            return "BOT_V7"
        elif ai_type == OpponentAIType.BOT_V8:
            return "BOT_V8"
        else:
            return "UNKNOWN"

    def print_ships_table(self, env, info: Optional[Dict] = None):
        """Print player and opponent ship details in a table sorted by credits (desc).

        Columns: Ship, Credits, Pos, State, Action, Payload, Energy, Health, Nutrinium, Status
        """
        # Collect rows for player and opponents
        rows = []

        # Player
        ps = getattr(env, 'player_ship', {})
        p_name = ps.get('name', 'PLAYER')
        p_credits = int(ps.get('credits', 0))
        p_action = (info.get('action') if info else '') or ''
        p_payload = (info.get('payload') if info else '') or ''
        p_energy = int(ps.get('energy', 0))
        p_health = int(ps.get('health', 0))
        p_nutr = int(ps.get('nutrinium', 0))
        p_status = 'DESTROYED' if ps.get('destroyed', False) else 'ALIVE'
        # Coordinates and state
        p_x = ps.get('x', '')
        p_y = ps.get('y', '')
        p_pos = f"({p_x},{p_y})" if p_x != '' and p_y != '' else ''
        p_state = ps.get('state', '')
        rows.append((p_name, p_credits, p_pos, p_state, str(p_action), str(p_payload), p_energy, p_health, p_nutr, p_status))

        # Opponents
        last_results = getattr(env, 'last_opponent_action_results', {}) or {}
        for i, enemy in enumerate(getattr(env, 'opponent_ships', [])):
            e_name = enemy.get('name', f'E{i+1}')
            e_credits = int(enemy.get('credits', 0))
            op_res = last_results.get(i, {})
            e_action = op_res.get('action') if op_res else ''
            e_payload = op_res.get('payload') if op_res else ''
            e_energy = int(enemy.get('energy', 0))
            e_health = int(enemy.get('health', 0))
            e_nutr = int(enemy.get('nutrinium', 0))
            e_status = 'DESTROYED' if enemy.get('destroyed', False) else 'ALIVE'
            e_x = enemy.get('x', '')
            e_y = enemy.get('y', '')
            e_pos = f"({e_x},{e_y})" if e_x != '' and e_y != '' else ''
            e_state = enemy.get('state', '')
            rows.append((e_name, e_credits, e_pos, e_state, str(e_action), str(e_payload), e_energy, e_health, e_nutr, e_status))

        # Sort by credits descending
        rows.sort(key=lambda r: r[1], reverse=True)

        # Prepare columns and widths
        headers = ["Ship", "Credits", "Pos", "State", "Action", "Payload", "Energy", "Health", "Nutrinium", "Status"]
        cols = list(zip(*([[str(h) for h in headers]] + [[str(v) for v in row] for row in rows])))
        widths = [max(len(cell) for cell in col) for col in cols]

        # Print header
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        sep_line = "  ".join('-' * w for w in widths)
        print("\n" + header_line)
        print(sep_line)

        # Print rows
        for row in rows:
            line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            print(line)

    def load_model(self, env):
        """Load the trained model for the given environment.

        Handles compatibility with old models trained with 14 actions
        by wrapping them in a compatibility layer.
        Also handles old models trained with flat Box observation space
        vs new Dict observation space (action masking).
        """
        from gymnasium import spaces

        # Try to load the model first to check its action/observation space
        try:
            if self.algorithm == "PPO":
                temp_model = PPO.load(self.model_path)
            elif self.algorithm == "DQN":
                temp_model = DQN.load(self.model_path)
            elif self.algorithm == "A2C":
                temp_model = A2C.load(self.model_path)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            # Native models trained with the current environment use a MultiDiscrete
            # action space ([action_type, target_x, target_y, energy_bin]) that matches
            # the env directly. Load them as-is - the compatibility adapter below only
            # supports legacy scalar Discrete models and cannot consume a vector action.
            if isinstance(temp_model.action_space, spaces.MultiDiscrete):
                # If the env observation is larger than what the model was trained on
                # (e.g. appended action-restriction features), the native PPO.load(env=env)
                # rejects the space mismatch. Wrap such models so the observation is
                # truncated to the model's expected size; otherwise load natively.
                model_obs_size = None
                env_obs_size = None
                if (isinstance(temp_model.observation_space, spaces.Dict) and
                        isinstance(env.observation_space, spaces.Dict) and
                        'observation' in temp_model.observation_space.spaces and
                        'observation' in env.observation_space.spaces):
                    model_obs_size = temp_model.observation_space['observation'].shape[0]
                    env_obs_size = env.observation_space['observation'].shape[0]
                if (model_obs_size is not None and env_obs_size is not None
                        and model_obs_size != env_obs_size):
                    print(f"\nWARNING: Model observation size ({model_obs_size}) != Environment observation size ({env_obs_size})")
                    print(f"Using observation-truncation wrapper for native MultiDiscrete model.\n")
                    from model_adapter import wrap_multidiscrete_model_with_obs_compat
                    return wrap_multidiscrete_model_with_obs_compat(temp_model)
                if self.algorithm == "PPO":
                    return PPO.load(self.model_path, env=env)
                elif self.algorithm == "DQN":
                    return DQN.load(self.model_path, env=env)
                else:  # A2C
                    return A2C.load(self.model_path, env=env)

            # Check if action spaces match. The env now uses a MultiDiscrete action space
            # whose first dimension is the number of action *types* (19). Legacy models are
            # scalar Discrete(14/15); their scalar output is still a valid action type in the
            # new space (action ids 0-13 are unchanged), so we accept them through the compat
            # wrapper which maps/masks model action ids into the env action-type set.
            model_action_space = getattr(temp_model.action_space, 'n', None)
            env_action_space = getattr(env, 'num_action_types', None)
            if env_action_space is None:
                env_action_space = (
                    int(env.action_space.nvec[0]) if hasattr(env.action_space, 'nvec')
                    else getattr(env.action_space, 'n', None)
                )

            needs_action_compat = False
            if model_action_space is None:
                raise ValueError(
                    "Model uses a structured (non-Discrete) action space; unsupported for legacy load")
            if model_action_space != env_action_space:
                print(f"\nWARNING: Model action space ({model_action_space}) != Environment action space ({env_action_space})")
                print(f"This model was trained with an older version of the environment.")

                if model_action_space <= env_action_space:
                    print(f"Detected: legacy {model_action_space}-action model vs current {env_action_space}-action environment")
                    print(f"Solution: Using compatibility mode - legacy action ids mapped into the new action set")
                    print(f"Note: For best results, retrain the model with the new action space\n")
                    needs_action_compat = True
                else:
                    raise ValueError(f"Incompatible action spaces: {model_action_space} vs {env_action_space}")

            # Check if model expects flat observation (Box) vs new Dict space
            model_obs_space = temp_model.observation_space
            needs_obs_compat = isinstance(model_obs_space, spaces.Box) and isinstance(env.observation_space, spaces.Dict)

            spec = getattr(self, 'player_model_spec', None)
            spec_name = str(getattr(spec, 'name', '')).lower()
            spec_requests_flat_box = bool(
                spec_name == 'default_full_box'
            )

            if needs_obs_compat:
                if not spec_requests_flat_box:
                    print(f"  Model expects flat observation (Box), environment provides Dict.")
                    print(f"  Using compatibility wrapper to extract flat observation.\n")

            # Check if model expects Dict obs with a different (smaller) observation size
            needs_obs_size_compat = False
            spec_handles_obs_size = False
            if (isinstance(model_obs_space, spaces.Dict) and
                    isinstance(env.observation_space, spaces.Dict) and
                    'observation' in model_obs_space.spaces and
                    'observation' in env.observation_space.spaces):
                model_obs_size = model_obs_space['observation'].shape[0]
                env_obs_size = env.observation_space['observation'].shape[0]
                if model_obs_size != env_obs_size:
                    # If the player_model_spec's observation size matches the model, the spec
                    # system will generate the correctly-sized observation - no compat needed.
                    spec_obs_size = spec.observation_spec.observation_size if spec else None
                    if spec_obs_size is not None and spec_obs_size == model_obs_size:
                        spec_handles_obs_size = True  # Spec system handles obs - use temp_model
                    else:
                        print(f"\nWARNING: Model observation size ({model_obs_size}) != Environment observation size ({env_obs_size})")
                        print(f"This model was trained with an older version of the environment.")
                        print(f"Using compatibility mode - observation will be truncated to {model_obs_size} features.\n")
                        needs_obs_size_compat = True

            # Use compatibility wrapper if needed for action, observation type, or obs size.
            # Also use temp_model (loaded without env) when the spec system handles obs sizing
            # to avoid the env obs-space mismatch error on PPO.load(env=env).
            if needs_action_compat or needs_obs_compat or needs_obs_size_compat or spec_handles_obs_size:
                return wrap_model_with_compat(
                    temp_model,
                    env_action_space_size=env_action_space,
                    enable_action_masking=True,
                )

            # Fully compatible model - still wrap so the env's action mask is
            # honoured at predict time (cheap no-op for models that already
            # pick valid actions, but it RESCUES converted MaskablePPO models
            # whose mask was discarded by `convert_models.py`).
            if self.algorithm == "PPO":
                base_model = PPO.load(self.model_path, env=env)
            elif self.algorithm == "DQN":
                base_model = DQN.load(self.model_path, env=env)
            elif self.algorithm == "A2C":
                base_model = A2C.load(self.model_path, env=env)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            return wrap_model_with_compat(
                base_model,
                env_action_space_size=env_action_space,
                enable_action_masking=True,
            )

        except Exception as e:
            # If loading fails, raise the error
            raise RuntimeError(f"Failed to load model: {e}")

    def run_episode(self,
                   num_opponents: int,
                   render: bool = True,
                   render_interval: int = 20,
                   deterministic: bool = True,
                   pause_each_step: bool = False,
                   print_all_actions: bool = False,
                   print_each_step: bool = False,
                   cell_width: Optional[int] = None,
                   minimap: bool = False,
                   minimap_radius: int = 3,
                   forced_opponent_types: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
        """
        Run a single episode with the trained model.

        Args:
            num_opponents: Number of opponent ships
            render: Whether to render the game
            render_interval: Steps between renders
            deterministic: Use deterministic policy (best action)
            pause_each_step: If True, pause and wait for spacebar after each render
            print_all_actions: If True, print action taken at every step (for debugging)

        Returns:
            (episode_stats, control)
            - episode_stats: Dictionary with episode statistics
            - control: dict with keys 'skip' (bool) and 'quit' (bool) to inform caller
        """
        # Create environment
        env = ProspectorsPiratesEnv(
            map_width=self.map_width,
            map_height=self.map_height,
            num_opponents=num_opponents,
            max_steps=self.max_steps,
            render_mode='human' if render else None,
            use_predefined_asteroids=self.use_predefined_asteroids,
            asteroid_config_path=self.asteroid_config_path,
            use_predefined_start=self.use_predefined_start,
            start_position_config_path=self.start_position_config_path,
            terminate_on_player_death=False,  # Allow game to continue after player death for full simulation
            cell_width=cell_width,
            minimap_mode=minimap,
            minimap_radius=minimap_radius,
            forced_opponent_types=forced_opponent_types
        )

        # Apply optional player MODEL_SPEC from --model-path (MODEL_PATH::SPEC_NAME)
        env.player_model_spec = self.player_model_spec

        # Load model and reset the environment
        model = self.load_model(env)
        observation, info = env.reset()

        # Print opponent AI types at start
        if print_each_step or pause_each_step:
            self._print_opponents(env)

        total_reward = 0
        done = False
        step = 0
        control = {'skip': False, 'quit': False}
        # Track per-participant kills for this episode.
        player_kills = 0
        enemy_kills_by_index = {i: 0 for i in range(len(env.opponent_ships))}

        while not done:
            action = self._predict_player_action(model, observation, deterministic, env)

            # Log/render periodically - BEFORE executing the action so the table
            # reflects the current (pre-action) state. Always log to file when a
            # logger is enabled, but only show in console when render=True.
            should_log_details = (self.logger is not None) or (render and step % render_interval == 0)
            show_step_tables = should_log_details and step % render_interval == 0

            predicted_action_name = None
            if show_step_tables:
                predicted_action_name = self._render_pre_action_table(env, action, step, render)

                # Pause if requested (before action execution)
                if pause_each_step and not control['skip'] and not control['quit']:
                    rv = self._wait_for_spacebar()
                    if rv == 'skip':
                        control['skip'] = True
                    elif rv == 'quit':
                        control['quit'] = True
                        # Set done to True to break out and finish this episode early
                        done = True

            # Execute the action
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Track kill events from this step's action payloads.
            player_kills, enemy_kills_by_index = self._track_step_kills(
                env, info, player_kills, enemy_kills_by_index
            )

            # Show action results (after execution)
            if show_step_tables:
                self._render_post_action_table(
                    env, info, total_reward, step, render, predicted_action_name
                )

            step += 1

        # Final render
        if render:
            env.render()
            if pause_each_step and not control['skip']:
                print("\n[Episode Complete - Press SPACE to continue or ESC to quit...]")
                rv = self._wait_for_spacebar()
                if rv == 'skip':
                    control['skip'] = True
                elif rv == 'quit':
                    control['quit'] = True

        episode_stats = self._build_episode_stats(
            env, info, step, total_reward, player_kills, enemy_kills_by_index, num_opponents
        )
        env.close()
        return episode_stats, control

    def run_simulation(self,
                       num_episodes: int = 5,
                       min_opponents: int = 1,
                       max_opponents: int = 5,
                       render: bool = True,
                       render_interval: int = 20,
                       verbose: bool = True,
                       pause_each_step: bool = False,
                       cell_width: Optional[int] = None,
                       minimap: bool = False,
                       minimap_radius: int = 3,
                       print_each_step: bool = False,
                       forced_opponent_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Run multiple episodes with random number of opponents.

        Args:
            num_episodes: Number of episodes to run
            min_opponents: Minimum number of opponents
            max_opponents: Maximum number of opponents
            render: Whether to render the game
            render_interval: Steps between renders
            verbose: Print detailed output
            pause_each_step: If True, pause and wait for spacebar after each render
            print_each_step: If True, print detailed info for each step during simulation

        Returns:
            List of episode statistics
        """
        # Set process title for Task Manager visibility
        if SETPROCTITLE_AVAILABLE:
            setproctitle("PnP Playing")
        # Set console window title (more visible on Windows)
        set_console_title("PnP Playing")

        if verbose:
            self._print("\n" + "=" * 70, to_episode=False)
            self._print("PROSPECTORS N PIRATES - Trained Model Simulation", to_episode=False)
            self._print(f"Model: {self.model_path}", to_episode=False)
            self._print(f"Algorithm: {self.algorithm}", to_episode=False)
            self._print(f"Episodes: {num_episodes}", to_episode=False)
            if forced_opponent_types:
                self._print(f"Opponents: {', '.join(forced_opponent_types)}", to_episode=False)
            else:
                self._print(f"Opponents: {min_opponents} to {max_opponents}", to_episode=False)
            if pause_each_step:
                self._print("Mode: Interactive (press SPACE to advance, Q to skip pauses, ESC to quit)", to_episode=False)
            if self.logger:
                self._print(f"Logging to: {self.logger.output_dir}", to_episode=False)
            self._print("=" * 70, to_episode=False)

        self.episode_stats = []
        skip_pauses = False

        for episode in range(num_episodes):
            # Start episode logging
            if self.logger:
                self.logger.start_episode(episode + 1, num_episodes)

            # Determine number of opponents for this episode
            if forced_opponent_types:
                num_opponents = len(forced_opponent_types)
            else:
                num_opponents = random.randint(min_opponents, max_opponents)

            if verbose:
                self._print(f"\n{'=' * 70}")
                self._print(f"EPISODE {episode + 1}/{num_episodes} - {num_opponents} opponent(s)")
                self._print(f"{'=' * 70}")

            # Run episode
            stats, control = self.run_episode(
                num_opponents=num_opponents,
                render=render,
                render_interval=render_interval,
                pause_each_step=pause_each_step and not skip_pauses,
                cell_width=cell_width,
                minimap=minimap,
                minimap_radius=minimap_radius,
                print_each_step=print_each_step,
                forced_opponent_types=forced_opponent_types
            )

            # Append stats collected so far
            self.episode_stats.append(stats)

            # If user chose to skip remaining pauses, update flag so subsequent episodes won't pause
            if control.get('skip'):
                skip_pauses = True

            # If user requested quit (ESC), stop the simulation early
            if control.get('quit'):
                if verbose:
                    self._print('\n[User requested to quit simulation early (ESC pressed).]')
                # End episode logging before breaking
                if self.logger:
                    self.logger.end_episode()
                break

            # Print episode results
            if verbose:
                self.print_episode_results(episode + 1, stats)

            # End episode logging
            if self.logger:
                self.logger.end_episode()

        # Print summary
        if verbose:
            self.print_summary()

        # Close logger
        if self.logger:
            self.logger.close()

        return self.episode_stats

    def print_episode_results(self, episode_num: int, stats: Dict):
        """Print detailed results for a single episode."""
        self._print(f"\n--- Episode {episode_num} Results ---")
        self._print(f"  Opponents: {stats['num_opponents']}")
        self._print(f"  Steps: {stats['steps']}")

        # Calculate player placement and update cumulative stats
        self.calculate_player_placement(stats)

        # Print cumulative placement stats
        self._print(f"  Player Placements (cumulative): 1st: {self.player_placements[1]}, 2nd: {self.player_placements[2]}, 3rd: {self.player_placements[3]}")
        # Podium Chance: % of episodes where player finished top 3
        _total_eps = len(self.episode_stats)
        _podium_count = self.player_placements[1] + self.player_placements[2] + self.player_placements[3]
        _podium_pct = (_podium_count / _total_eps * 100) if _total_eps > 0 else 0.0
        self._print(f"  Podium Chance: {_podium_pct:.1f}% ({_podium_count}/{_total_eps} episodes in top 3)")

        # Print episode summary table (player + enemies)
        try:
            self.print_episode_table(stats)
        except Exception:
            # Fallback: ignore table errors and continue with existing prints
            pass

        # Individual enemy stats
        # for i, enemy in enumerate(stats['enemy_details']):
        #     # Get ship name if available (should be available from environment)
        #     enemy_name = enemy.get('name', f'E{i+1}')
        #     ai_type = enemy.get('ai_type', 0)
        #     ai_type_name = self.get_ai_type_name(ai_type)
        #
        #     # For MODEL type, append model name
        #     if ai_type == OpponentAIType.MODEL:
        #         model_path = enemy.get('model_path', 'unknown')
        #         model_name = os.path.basename(model_path) if model_path else 'unknown'
        #         ai_display = f"{ai_type_name}:{model_name}"
        #     else:
        #         ai_display = ai_type_name
        #
        #     status = "DESTROYED" if enemy['destroyed'] else "ALIVE"
        #     print(f"  {enemy_name} ({ai_display}): {status}, HP:{enemy['health']}, "
        #           f"Credits:{enemy['credits']}, Nutrinium:{enemy['nutrinium']}")
        # # Print enemy abilities/skills horizontally
        # e_abilities = enemy.get('abilities', {}) or {}
        # if e_abilities:
        #     e_items = ", ".join(f"{k}:{v}" for k, v in sorted(e_abilities.items()))
        #     print(f"  Abilities: {e_items}")

    def calculate_player_placement(self, stats: Dict):
        """Calculate player's placement (1st, 2nd, 3rd) for this episode and update cumulative stats."""
        # Collect all participants with their ranking keys.
        participants = []

        # Player
        participants.append({
            'name': 'PLAYER',
            'credits': stats.get('player_credits', 0) or 0,
            'nutrinium': stats.get('player_nutrinium', 0) or 0,
            'kills': stats.get('player_kills', 0) or 0,
            'energy': stats.get('player_energy', 0) or 0,
            'is_player': True
        })

        # Enemies
        for enemy in stats.get('enemy_details', []):
            participants.append({
                'name': enemy.get('name', 'ENEMY'),
                'credits': enemy.get('credits', 0) or 0,
                'nutrinium': enemy.get('nutrinium', 0) or 0,
                'kills': enemy.get('kills', 0) or 0,
                'energy': enemy.get('energy', 0) or 0,
                'is_player': False
            })

        # Rank: credits desc, then nutrinium desc, then ships destroyed (kills)
        # desc, then energy ascending.
        participants.sort(key=lambda p: (-p['credits'], -p['nutrinium'], -p['kills'], p['energy']))

        # Find player's placement (1-based)
        for i, p in enumerate(participants):
            if p['is_player']:
                placement = i + 1
                # Only track top 3 placements
                if placement <= 3:
                    self.player_placements[placement] += 1
                break


    def print_episode_table(self, stats: Dict):
        """Print a compact table summarizing the episode results: player and enemies.

        Columns: Ship, Role, Credits, Health, Nutrinium, Energy, Status, Abilities
        Sorted by Credits descending.
        """
        entries = []

        # Player row
        entries.append({
            'name': 'PLAYER',
            'role': 'PLAYER',
            'credits': int(stats.get('player_credits', 0) or 0),
            'health': int(stats.get('player_health', 0) or 0),
            'nutr': int(stats.get('player_nutrinium', 0) or 0),
            'energy': int(stats.get('player_energy', 0)) if stats.get('player_energy') is not None else '',
            'kills': int(stats.get('player_kills', 0) or 0),
            'energy_val': int(stats.get('player_energy', 0) or 0),
            'status': 'DESTROYED' if stats.get('player_destroyed', False) else 'ALIVE',
            'abilities': stats.get('player_abilities') or {},
        })

        # Enemies
        for enemy in stats.get('enemy_details', []):
            ai_type = enemy.get('ai_type', None)
            e_role = self.get_ai_type_name(ai_type) if ai_type is not None else 'ENEMY'
            if ai_type == OpponentAIType.MODEL:
                model_path = enemy.get('model_path', 'unknown')
                model_name = os.path.basename(model_path) if model_path else 'unknown'
                e_role = f"{e_role}:{model_name}"
            entries.append({
                'name': enemy.get('name', 'ENEMY'),
                'role': e_role,
                'credits': int(enemy.get('credits', 0) or 0),
                'health': int(enemy.get('health', 0) or 0),
                'nutr': int(enemy.get('nutrinium', 0) or 0),
                'energy': '',
                'kills': int(enemy.get('kills', 0) or 0),
                'energy_val': int(enemy.get('energy', 0) or 0),
                'status': 'DESTROYED' if enemy.get('destroyed', False) else 'ALIVE',
                'abilities': enemy.get('abilities') or {},
            })

        # Rank: credits desc, then nutrinium desc, then ships destroyed (kills)
        # desc, then energy ascending.
        entries.sort(key=lambda e: (-e['credits'], -e['nutr'], -e['kills'], e['energy_val']))

        # Build each ship's abilities as a comma-separated "skill=value" list,
        # ordered by skill value descending so the strongest skills read first.
        for e in entries:
            e['ability_tokens'] = [
                f"{k}={v}"
                for k, v in sorted(e['abilities'].items(), key=lambda kv: (-kv[1], kv[0]))
                if v and v > 0
            ]

        # Fixed (non-wrapping) columns come first; Abilities is the last column and
        # wraps onto indented continuation lines when it would exceed the line width.
        fixed_headers = ["Ship", "Role", "Credits", "Health", "Nutrinium", "Energy", "Status"]
        fixed_rows = [
            (e['name'], e['role'], e['credits'], e['health'], e['nutr'], e['energy'], e['status'])
            for e in entries
        ]
        cols = list(zip(*([[str(h) for h in fixed_headers]] + [[str(v) for v in row] for row in fixed_rows])))
        widths = [max(len(cell) for cell in col) for col in cols]

        # Column offset where the Abilities text begins, so continuation lines align
        # under the Abilities header: the fixed columns joined by "  " plus the "  "
        # separator before the Abilities column == sum(widths) + 2 * len(widths).
        indent = sum(widths) + 2 * len(widths)

        # Wrap abilities to fit the remaining line width, breaking only at commas
        # (never splitting a "skill=value" token).
        import shutil
        max_line_width = shutil.get_terminal_size((120, 20)).columns
        available = max(20, max_line_width - indent)

        def _wrap_abilities(tokens):
            if not tokens:
                return ['']
            lines = []
            current = ''
            for tok in tokens:
                candidate = tok if not current else f"{current}, {tok}"
                if current and len(candidate) > available:
                    lines.append(current)
                    current = tok
                else:
                    current = candidate
            lines.append(current)
            return lines

        wrapped = [_wrap_abilities(e['ability_tokens']) for e in entries]
        abilities_width = max(
            [len("Abilities")] + [len(line) for lines in wrapped for line in lines]
        )

        # Prepare header and separator
        header_cells = [h.ljust(w) for h, w in zip(fixed_headers, widths)] + ["Abilities".ljust(abilities_width)]
        header_line = "  ".join(header_cells)
        sep_line = "  ".join('-' * w for w in widths + [abilities_width])
        self._print("\n" + header_line)
        self._print(sep_line)

        # Print rows; abilities wrap onto indented continuation lines aligned under
        # the Abilities header.
        for row, ability_lines in zip(fixed_rows, wrapped):
            prefix = "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            self._print((prefix + "  " + ability_lines[0]).rstrip())
            for cont in ability_lines[1:]:
                self._print(" " * indent + cont)

    def print_summary(self):
        """Print summary statistics across all episodes."""
        if not self.episode_stats:
            self._print("No episodes run yet.", to_episode=False)
            return

        # Aggregate statistics
        rewards = [s['total_reward'] for s in self.episode_stats]
        credits = [s['player_credits'] for s in self.episode_stats]
        steps = [s['steps'] for s in self.episode_stats]
        destroyed = [s['player_destroyed'] for s in self.episode_stats]
        enemies_destroyed = [s['enemies_destroyed'] for s in self.episode_stats]

        self._print("\n" + "=" * 70, to_episode=False)
        self._print("SIMULATION SUMMARY", to_episode=False)
        self._print("=" * 70, to_episode=False)
        self._print(f"Episodes Completed: {len(self.episode_stats)}", to_episode=False)
        self._print(f"Average Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}", to_episode=False)
        self._print(f"Average Credits: {np.mean(credits):.1f} +/- {np.std(credits):.1f}", to_episode=False)
        self._print(f"Average Steps: {np.mean(steps):.1f}", to_episode=False)
        self._print(f"Max Credits: {max(credits)}", to_episode=False)
        self._print(f"Player Survival Rate: {(1 - np.mean(destroyed)) * 100:.1f}%", to_episode=False)
        self._print(f"Avg Enemies Destroyed: {np.mean(enemies_destroyed):.2f}", to_episode=False)
        self._print(f"Total Enemies Destroyed: {sum(enemies_destroyed)}", to_episode=False)
        # Print the player's model information used for the simulation
        try:
            model_display = os.path.basename(self.model_path) if self.model_path else 'unknown'
        except Exception:
            model_display = str(self.model_path)
        self._print(f"Player Model: {model_display}", to_episode=False)
        self._print("=" * 70, to_episode=False)

        # Participant stats across all episodes
        self._print_participant_stats()

    def get_statistics(self) -> Dict:
        """
        Get aggregated statistics from all episodes.

        Returns:
            Dictionary with summary statistics
        """
        if not self.episode_stats:
            return {}

        rewards = [s['total_reward'] for s in self.episode_stats]
        credits = [s['player_credits'] for s in self.episode_stats]
        steps = [s['steps'] for s in self.episode_stats]
        destroyed = [s['player_destroyed'] for s in self.episode_stats]
        enemies_destroyed = [s['enemies_destroyed'] for s in self.episode_stats]

        return {
            'num_episodes': len(self.episode_stats),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_credits': np.mean(credits),
            'std_credits': np.std(credits),
            'max_credits': max(credits),
            'avg_steps': np.mean(steps),
            'survival_rate': 1 - np.mean(destroyed),
            'avg_enemies_destroyed': np.mean(enemies_destroyed),
            'total_enemies_destroyed': sum(enemies_destroyed)
        }

    # =================================
    # Private helpers
    # =================================

    def _print_participant_stats(self):
        """Print aggregated stats for each participant across all episodes."""
        if not self.episode_stats:
            return

        num_episodes = len(self.episode_stats)

        # Aggregate per-participant stats
        # Key: participant identifier (PLAYER or enemy name+type)
        participants = {}

        for stats in self.episode_stats:
            # First, rank all participants in this episode by credits
            episode_rankings = []

            # Player
            p_credits = stats.get('player_credits', 0) or 0
            p_nutrinium = stats.get('player_nutrinium', 0) or 0
            episode_rankings.append(('PLAYER', p_credits, p_nutrinium))

            # Enemies
            for enemy in stats.get('enemy_details', []):
                e_name = enemy.get('name', 'ENEMY')
                e_credits = enemy.get('credits', 0) or 0
                e_nutrinium = enemy.get('nutrinium', 0) or 0
                episode_rankings.append((e_name, e_credits, e_nutrinium))

            # Sort by credits descending, then nutrinium descending to break ties
            # (mirrors calculate_player_placement so PLAYER isn't favored on credit ties)
            episode_rankings.sort(key=lambda x: (x[1], x[2]), reverse=True)

            # Create placement map: participant_name -> placement (1, 2, 3, ...)
            placements = {name: rank + 1 for rank, (name, _, _) in enumerate(episode_rankings)}

            # Now accumulate stats for player
            p_key = 'PLAYER'
            if p_key not in participants:
                participants[p_key] = {
                    'name': 'PLAYER', 'role': 'PLAYER',
                    'total_credits': 0, 'max_credits': 0,
                    'survived': 0, 'episodes': 0,
                    'total_enemies_destroyed': 0,
                    'total_nutrinium': 0, 'total_health': 0,
                    'placements': {},  # placement -> count
                }
            p = participants[p_key]
            p['episodes'] += 1
            p_credits = stats.get('player_credits', 0) or 0
            p['total_credits'] += p_credits
            p['max_credits'] = max(p['max_credits'], p_credits)
            if not stats.get('player_destroyed', False):
                p['survived'] += 1
            p['total_enemies_destroyed'] += stats.get('player_kills', stats.get('enemies_destroyed', 0))
            p['total_nutrinium'] += stats.get('player_nutrinium', 0) or 0
            p['total_health'] += stats.get('player_health', 0) or 0
            # Track placement
            placement = placements.get('PLAYER', 0)
            p['placements'][placement] = p['placements'].get(placement, 0) + 1

            # Enemies
            for enemy in stats.get('enemy_details', []):
                e_name = enemy.get('name', 'ENEMY')
                ai_type = enemy.get('ai_type', None)
                role = self.get_ai_type_name(ai_type) if ai_type is not None else 'ENEMY'
                if ai_type == OpponentAIType.MODEL:
                    model_path = enemy.get('model_path', 'unknown')
                    model_name = os.path.basename(model_path) if model_path else 'unknown'
                    role = f"MODEL:{model_name}"

                e_key = e_name
                if e_key not in participants:
                    participants[e_key] = {
                        'name': e_name, 'role': role,
                        'total_credits': 0, 'max_credits': 0,
                        'survived': 0, 'episodes': 0,
                        'total_enemies_destroyed': 0,
                        'total_nutrinium': 0, 'total_health': 0,
                        'placements': {},  # placement -> count
                    }
                e = participants[e_key]
                e['episodes'] += 1
                e_credits = enemy.get('credits', 0) or 0
                e['total_credits'] += e_credits
                e['max_credits'] = max(e['max_credits'], e_credits)
                if not enemy.get('destroyed', False):
                    e['survived'] += 1
                e['total_enemies_destroyed'] += enemy.get('kills', 0) or 0
                e['total_nutrinium'] += enemy.get('nutrinium', 0) or 0
                e['total_health'] += enemy.get('health', 0) or 0
                # Track placement
                placement = placements.get(e_name, 0)
                e['placements'][placement] = e['placements'].get(placement, 0) + 1

        # Build rows with all stats
        rows = []
        for key, p in participants.items():
            eps = p['episodes']
            avg_credits = p['total_credits'] / eps if eps else 0
            survival_rate = (p['survived'] / eps * 100) if eps else 0
            avg_nutr = p['total_nutrinium'] / eps if eps else 0
            avg_health = p['total_health'] / eps if eps else 0

            # Format placements: show 1st, 2nd, 3rd counts
            first_place = p['placements'].get(1, 0)
            second_place = p['placements'].get(2, 0)
            third_place = p['placements'].get(3, 0)
            podium_count = first_place + second_place + third_place
            podium_pct = (podium_count / eps * 100) if eps > 0 else 0.0

            rows.append({
                'name': p['name'],
                'role': p['role'],
                'avg_credits': avg_credits,
                'max_credits': p['max_credits'],
                'survival_rate': survival_rate,
                'total_enemies_destroyed': p['total_enemies_destroyed'],
                'avg_nutr': avg_nutr,
                'avg_health': avg_health,
                'first_place': first_place,
                'second_place': second_place,
                'third_place': third_place,
                'podium_pct': podium_pct,
            })

        # Sort by: 1st place (desc), 2nd place (desc), 3rd place (desc), avg credits (desc), avg nutr (desc)
        rows.sort(key=lambda r: (
            -r['first_place'],      # Most 1st place wins first
            -r['second_place'],     # Then most 2nd place
            -r['third_place'],      # Then most 3rd place
            -r['avg_credits'],      # Then highest avg credits
            -r['avg_nutr']          # Then highest avg nutrinium
        ))

        # Add rank and format for display
        ranked_rows = []
        for i, row in enumerate(rows):
            ranked_rows.append((
                str(i + 1),
                row['name'],
                row['role'],
                f"{row['avg_credits']:.1f}",
                str(row['max_credits']),
                f"{row['survival_rate']:.0f}%",
                str(row['total_enemies_destroyed']),
                f"{row['avg_nutr']:.1f}",
                f"{row['avg_health']:.0f}",
                str(row['first_place']),
                str(row['second_place']),
                str(row['third_place']),
                f"{row['podium_pct']:.0f}%",
            ))

        headers = ["Rank", "Ship", "Role", "Avg Credits", "Max Credits",
                   "Survival%", "Enemies Killed", "Avg Nutrinium", "Avg Health",
                   "1st", "2nd", "3rd", "Podium%"]
        all_rows = [headers] + ranked_rows
        cols = list(zip(*all_rows))
        widths = [max(len(str(cell)) for cell in col) for col in cols]

        self._print("\n" + "=" * 70, to_episode=False)
        self._print("PARTICIPANT STATS", to_episode=False)
        self._print("=" * 70, to_episode=False)

        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        sep_line = "  ".join('-' * w for w in widths)
        self._print(header_line, to_episode=False)
        self._print(sep_line, to_episode=False)
        for row in ranked_rows:
            line = " ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            self._print(line, to_episode=False)
        self._print("=" * 70, to_episode=False)

    def _print(self, message: str = "", to_episode: bool = True, to_simulation: bool = True, to_console: bool = True):
        """
        Print message to console and optionally to log files.

        Args:
            message: Message to print/log
            to_episode: Whether to write to current episode log
            to_simulation: Whether to write to simulation log
            to_console: Whether to print to console
        """
        if self.logger:
            self.logger.log(message, to_episode=to_episode, to_simulation=to_simulation, to_console=to_console)
        else:
            if to_console:
                print(message)

    def _log_episode_detail(self, message: str = "", render: bool = True):
        """
        Log episode detail message. Always logs to episode file, but only prints to console if render=True.

        Args:
            message: Message to log
            render: Whether currently rendering (controls console output)
        """
        if self.logger:
            # Always log to episode file, but only to console if rendering
            self.logger.log(message, to_episode=True, to_simulation=False, to_console=render)
        else:
            # No logger - only print if rendering
            if render:
                print(message)

    def _wait_for_spacebar(self):
        """Wait for user to press spacebar to continue, Q to skip future pauses, or ESC to quit.

        Returns:
            'continue' -> user pressed SPACE
            'skip' -> user pressed 'q' to skip remaining pauses
            'quit' -> user pressed ESC to quit the simulation
        """
        import msvcrt  # Windows-specific

        print("[Press SPACE to continue, Q to skip remaining pauses, ESC to quit...]", end='', flush=True)
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # Check for spacebar (0x20)
                if key == b' ':
                    print("\r" + " " * 80 + "\r", end='', flush=True)  # Clear the message
                    return 'continue'
                # Allow 'q' to skip remaining pauses
                elif key.lower() == b'q':
                    print("\r[Skipping remaining pauses...]" + " " * 20)
                    return 'skip'
                # ESC to quit (0x1b)
                elif key == b'\x1b':
                    print("\r[Quit requested - exiting simulation...]" + " " * 20)
                    return 'quit'
                # Some terminals may emit special prefix bytes (like 0 or 224) for arrows; ignore them
                elif key in (b'\x00', b'\xe0'):
                    # consume the next byte if present
                    if msvcrt.kbhit():
                        _ = msvcrt.getch()
                    continue
        # fallback
        return 'continue'

    def _print_opponents(self, env):
        """Print the opponent roster (name + AI type) at the start of an episode."""
        print(f"\n{'='*60}")
        print("OPPONENTS:")
        for i, enemy in enumerate(env.opponent_ships):
            e_name = enemy.get('name', f'E{i+1}')
            ai_type = enemy.get('ai_type', OpponentAIType.BOT_V2)
            ai_type_name = self.get_ai_type_name(ai_type)

            # If MODEL type, show the model path
            if ai_type == OpponentAIType.MODEL:
                model_path = enemy.get('model_path', 'unknown')
                # Extract just the model name from the path
                model_name = os.path.basename(model_path) if model_path else 'unknown'
                print(f"  {e_name}: {ai_type_name} ({model_name})")
            else:
                print(f"  {e_name}: {ai_type_name}")
        print(f"{'='*60}\n")

    def _predict_player_action(self, model, observation, deterministic, env) -> int:
        """Return the player's action id for this step.

        Returns RESPAWN (13) when the player ship is destroyed; otherwise queries
        the model and coerces the (possibly numpy) prediction into a plain int.
        """
        # Check if player is destroyed - if so, use RESPAWN action
        if env.player_ship.get('destroyed', False):
            return 13  # RESPAWN action

        action_raw, _ = model.predict(observation, deterministic=deterministic)

        # Defensive: convert numpy array/scalar action to int
        if isinstance(action_raw, np.ndarray):
            try:
                return int(action_raw.item())
            except Exception:
                return int(action_raw[0])
        try:
            return int(action_raw)
        except Exception:
            return action_raw

    def _compute_enforced_action_name(self, env, action, predicted_action_name) -> str:
        """Return the player's action display name, accounting for env enforcement.

        If ``action`` is invalid for the player's current state, this mirrors the
        enforcement logic in ``pnp_env.step()`` and returns "<enforced> (was <orig>)".
        Otherwise ``predicted_action_name`` is returned unchanged.
        """
        p = env.player_ship

        # Check if the predicted action is valid; if not, show what will actually execute
        is_valid, _reason = env._is_action_valid_for_state(action, p, is_player=True)
        if is_valid:
            return predicted_action_name

        # Mirror the enforcement logic in pnp_env.step()
        from pnp_env import ActionType as _AT
        if p.get('recharging', False):
            if p['energy'] >= env.config['max_energy']:
                enforced = 'RECHARGE_END'
            elif action not in (int(_AT.WAIT), int(_AT.RECHARGE_END)):
                # Model wants an active action -> end recharge
                enforced = 'RECHARGE_END'
            else:
                enforced = 'WAIT'
        elif p.get('destroyed', False):
            enforced = 'RESPAWN'
        else:
            # Pick best valid action from mask (mirrors pnp_env fallback)
            mask = env._get_action_mask(p)
            enforced = 'WAIT'
            # When energy is very low, prioritize RECHARGE
            if p['energy'] <= env.config['energy_costs'].get('move', 5):
                fb_order = [_AT.RECHARGE, _AT.MINE, _AT.SELL, _AT.WAIT,
                            _AT.JUMP_TO_ASTEROID,
                            _AT.JUMP_TO_TRADING_POST,
                            _AT.MOVE_NORTH, _AT.MOVE_SOUTH,
                            _AT.MOVE_EAST, _AT.MOVE_WEST,
                            _AT.ATTACK, _AT.RAISE_SHIELDS]
            else:
                fb_order = [_AT.MINE, _AT.SELL, _AT.JUMP_TO_ASTEROID,
                            _AT.JUMP_TO_TRADING_POST,
                            _AT.MOVE_NORTH, _AT.MOVE_SOUTH,
                            _AT.MOVE_EAST, _AT.MOVE_WEST,
                            _AT.RECHARGE, _AT.ATTACK, _AT.RAISE_SHIELDS,
                            _AT.WAIT]
            for fb in fb_order:
                if mask[int(fb)] == 1:
                    enforced = _AT(fb).name
                    break
        return f"{enforced} (was {predicted_action_name})"

    def _render_pre_action_table(self, env, action, step, render) -> str:
        """Render the pre-action map + 'CURRENT STATE & ACTIONS' table.

        Returns the player's (possibly enforcement-adjusted) action display name
        so the post-action table can reference it.
        """
        # 1. SHOW MAP FIRST (reflects previous step's action results) - only if rendering to console
        if render:
            env.render()

        # 2. SHOW CURRENT STATE AND ACTION IN TABLE FORMAT
        self._log_episode_detail(f"\n{'=' * 70}", render)
        self._log_episode_detail(f"CURRENT STATE & ACTIONS (Step {step})", render)
        self._log_episode_detail(f"{'=' * 70}", render)

        # Collect all ships (player + enemies) for table
        ships_data = []

        # Player
        p = env.player_ship
        player_state = p.get('state', 'READY')
        player_flags = []
        if p.get('shields_up'):
            player_flags.append('SHIELDS')
        if p.get('recharging'):
            player_flags.append('RECHARGING')
        player_flags_str = f"[{','.join(player_flags)}]" if player_flags else ''

        predicted_action_name = self.ACTION_NAMES[action] if action < len(self.ACTION_NAMES) else f"ACTION_{action}"
        predicted_action_name = self._compute_enforced_action_name(env, action, predicted_action_name)

        ships_data.append({
            'name': p.get('name', 'P'),
            'credits': p['credits'],
            'nutrinium': p['nutrinium'],
            'pos': f"({p['x']},{p['y']})",
            'energy': f"{p['energy']}/{env.config['max_energy']}",
            'health': f"{p['health']}/{env.config['max_health']}",
            'state': player_state + (' ' + player_flags_str if player_flags_str else ''),
            'action': predicted_action_name,
            'is_player': True
        })

        # Enemies - get their predicted next action
        for i, opp in enumerate(env.opponent_ships):
            if not opp.get('destroyed', False):
                opp_state = opp.get('state', 'READY')
                opp_flags = []
                if opp.get('shields_up'):
                    opp_flags.append('SHIELDS')
                if opp.get('recharging'):
                    opp_flags.append('RECHARGING')
                opp_flags_str = f"[{','.join(opp_flags)}]" if opp_flags else ''

                # Predict enemy action (call their AI to see what they'll do)
                try:
                    enemy_action = env._get_opponent_action(opp)
                    enemy_action_name = self.ACTION_NAMES[enemy_action] if enemy_action < len(self.ACTION_NAMES) else f"ACTION_{enemy_action}"
                except Exception:
                    enemy_action_name = '?'

                ships_data.append({
                    'name': opp.get('name', f'E{i+1}'),
                    'credits': opp.get('credits', 0),
                    'nutrinium': opp.get('nutrinium', 0),
                    'pos': f"({opp['x']},{opp['y']})",
                    'energy': f"{opp.get('energy', 0)}/{env.config['max_energy']}",
                    'health': f"{opp.get('health', 100)}/{env.config['max_health']}",
                    'state': opp_state + (' ' + opp_flags_str if opp_flags_str else ''),
                    'action': enemy_action_name,
                    'is_player': False
                })

        # Sort by credits (desc), then nutrinium (desc)
        ships_data.sort(key=lambda s: (s['credits'], s['nutrinium']), reverse=True)

        # Print table
        headers = ['Ship', 'Credits', 'Nutr', 'Pos', 'Energy', 'Health', 'State', 'Next Action']

        # Calculate column widths
        widths = [len(h) for h in headers]
        for ship in ships_data:
            widths[0] = max(widths[0], len(str(ship['name'])))
            widths[1] = max(widths[1], len(str(ship['credits'])))
            widths[2] = max(widths[2], len(str(ship['nutrinium'])))
            widths[3] = max(widths[3], len(ship['pos']))
            widths[4] = max(widths[4], len(ship['energy']))
            widths[5] = max(widths[5], len(ship['health']))
            widths[6] = max(widths[6], len(ship['state']))
            widths[7] = max(widths[7], len(ship['state']))

        # Print header
        header_line = ' '.join(h.ljust(w) for h, w in zip(headers, widths))
        sep_line = ' '.join('-' * w for w in widths)
        self._log_episode_detail('\n' + header_line, render)
        self._log_episode_detail(sep_line, render)

        # Print rows
        for ship in ships_data:
            row = [
                str(ship['name']).ljust(widths[0]),
                str(ship['credits']).ljust(widths[1]),
                str(ship['nutrinium']).ljust(widths[2]),
                ship['pos'].ljust(widths[3]),
                ship['energy'].ljust(widths[4]),
                ship['health'].ljust(widths[5]),
                ship['state'].ljust(widths[6]),
                ship['action'].ljust(widths[7])
            ]
            line = '  '.join(row)
            # Highlight player row
            if ship['is_player']:
                self._log_episode_detail(f"? {line}", render)
            else:
                self._log_episode_detail(f"  {line}", render)

        self._log_episode_detail("", render)

        return predicted_action_name

    def _track_step_kills(self, env, info, player_kills, enemy_kills_by_index):
        """Update and return (player_kills, enemy_kills_by_index) from this step.

        Kills are inferred from ATTACK action payloads containing destroyed=True,
        for both the player and each opponent.
        """
        # Player kill: determined by ATTACK payload containing destroyed=True.
        if info.get('action') == 'ATTACK':
            payload = info.get('payload')
            if isinstance(payload, dict) and payload.get('destroyed', False):
                player_kills += 1

        # Opponent kills: read per-opponent action results captured by env.step().
        last_enemy_results_for_kills = getattr(env, 'last_opponent_action_results', {}) or {}
        for idx, opp_result in last_enemy_results_for_kills.items():
            if not opp_result:
                continue
            if opp_result.get('action') != 'ATTACK':
                continue
            opp_payload = opp_result.get('payload')
            if isinstance(opp_payload, dict) and opp_payload.get('destroyed', False):
                enemy_kills_by_index[idx] = enemy_kills_by_index.get(idx, 0) + 1

        return player_kills, enemy_kills_by_index

    def _render_post_action_table(self, env, info, total_reward, step, render, predicted_action_name):
        """Render the post-action 'ACTION RESULTS' table for player + opponents."""
        self._log_episode_detail(f"\n{'=' * 70}", render)
        self._log_episode_detail(f"ACTION RESULTS (Step {step})", render)
        self._log_episode_detail(f"\n{'=' * 70}", render)

        # Collect action results for all ships
        action_results = []

        # Player result
        act_name = info.get('action', predicted_action_name)
        success = info.get('success', None)
        raw_r = info.get('raw_reward', None)
        scaled_r = info.get('scaled_reward', None)
        state_valid_flag = info.get('state_valid', True)
        payload = info.get('payload')

        result_str = 'OK' if success is True else ('FAIL' if success is False else '-')
        reward_str = f"{raw_r:+.2f}/{scaled_r:+.2f}" if raw_r is not None and scaled_r is not None else '-'

        # Format payload - use special formatting for ATTACK and MINE actions
        if act_name == 'ATTACK' and payload and isinstance(payload, dict) and 'target' in payload:
            payload_str = self._format_attack_details(payload)
        elif act_name == 'MINE' and payload and isinstance(payload, dict) and 'ast_mass' in payload:
            payload_str = self._format_mine_details(payload)
        else:
            payload_str = str(payload) if payload else ''

        # Truncate long payload strings
        if len(payload_str) > 250:
            payload_str = payload_str[:247] + '...'

        action_results.append({
            'name': env.player_ship.get('name', 'P'),
            'action': act_name,
            'result': result_str,
            'reward': reward_str,
            'details': payload_str,
            'is_player': True
        })

        # Enemy results
        last_enemy_results = getattr(env, 'last_opponent_action_results', {})
        last_enemy_actions = getattr(env, 'last_opponent_actions', {})

        for i, opp in enumerate(env.opponent_ships):
            if opp.get('destroyed', False):
                continue

            opp_action_id = last_enemy_actions.get(i)
            opp_action_name = self.ACTION_NAMES[opp_action_id] if opp_action_id is not None and opp_action_id < len(self.ACTION_NAMES) else '-'

            opp_result = last_enemy_results.get(i, {})
            opp_success = opp_result.get('success', None)
            opp_payload = opp_result.get('payload')

            opp_result_str = 'OK' if opp_success is True else ('FAIL' if opp_success is False else '-')

            # Format payload - use special formatting for ATTACK and MINE actions
            if opp_action_name == 'ATTACK' and opp_payload and isinstance(opp_payload, dict) and 'target' in opp_payload:
                opp_payload_str = self._format_attack_details(opp_payload)
            elif opp_action_name == 'MINE' and opp_payload and isinstance(opp_payload, dict) and 'ast_mass' in opp_payload:
                opp_payload_str = self._format_mine_details(opp_payload)
            else:
                opp_payload_str = str(opp_payload) if opp_payload else ''

            # Truncate long payload strings
            if len(opp_payload_str) > 250:
                opp_payload_str = opp_payload_str[:247] + '...'

            action_results.append({
                'name': opp.get('name', f'E{i+1}'),
                'action': opp_action_name,
                'result': opp_result_str,
                'reward': '-', # Don't show enemy rewards
                'details': opp_payload_str,
                'is_player': False
            })

        # Print table
        headers = ['Ship', 'Action', 'Result', 'Reward (raw/scaled)', 'Details']

        # Calculate column widths
        widths = [len(h) for h in headers]
        for ar in action_results:
            widths[0] = max(widths[0], len(str(ar['name'])))
            widths[1] = max(widths[1], len(ar['action']))
            widths[2] = max(widths[2], len(ar['result']))
            widths[3] = max(widths[3], len(ar['reward']))
            widths[4] = max(widths[4], len(ar['details']))

        # Print header
        header_line = ' '.join(h.ljust(w) for h, w in zip(headers, widths))
        sep_line = ' '.join('-' * w for w in widths)
        self._log_episode_detail('\n' + header_line, render)
        self._log_episode_detail(sep_line, render)

        # Print rows
        for ar in action_results:
            row = [
                str(ar['name']).ljust(widths[0]),
                ar['action'].ljust(widths[1]),
                ar['result'].ljust(widths[2]),
                ar['reward'].ljust(widths[3]),
                ar['details'].ljust(widths[4])
            ]
            line = ' '.join(row)
            # Highlight player row
            if ar['is_player']:
                self._log_episode_detail(f"? {line}", render)
            else:
                self._log_episode_detail(f"  {line}", render)

        # Show total reward for player
        self._log_episode_detail(f"\nPlayer Total Reward: {total_reward:.3f}", render)

        # Show state validation warning if needed
        if not state_valid_flag:
            self._log_episode_detail(f"WARN Warning: Action state invalid - {info.get('state_invalid_reason', '')}", render)

    def _build_episode_stats(self, env, info, step, total_reward, player_kills, enemy_kills_by_index, num_opponents) -> Dict:
        """Assemble the per-episode statistics dictionary from the final env state."""
        enemies_alive = sum(1 for e in env.opponent_ships if not e['destroyed'])
        enemies_destroyed = num_opponents - enemies_alive
        total_enemy_credits = sum(e['credits'] for e in env.opponent_ships)
        total_enemy_nutrinium = sum(e['nutrinium'] for e in env.opponent_ships if not e['destroyed'])
        avg_enemy_health = np.mean([e['health'] for e in env.opponent_ships if not e['destroyed']]) if enemies_alive > 0 else 0

        return {
            'num_opponents': num_opponents,
            'steps': step,
            'total_reward': total_reward,
            'player_credits': info.get('player_credits', 0),
            'player_nutrinium': info.get('player_nutrinium', 0),
            'player_health': info.get('player_health', 0),
            'player_energy': info.get('player_energy', 0),
            'player_destroyed': info.get('player_destroyed', False),
            'player_kills': player_kills,
            # Include player's abilities/skills and skill points from environment
            'player_abilities': dict(env.player_ship.get('abilities', {})) if getattr(env, 'player_ship', None) else {},
            'player_skill_points_total': env.player_ship.get('skill_points_total', None) if getattr(env, 'player_ship', None) else None,
            'player_skill_points_spent': env.player_ship.get('skill_points_spent', None) if getattr(env, 'player_ship', None) else None,
            'enemies_alive': enemies_alive,
            'enemies_destroyed': enemies_destroyed,
            'total_enemy_credits': total_enemy_credits,
            'total_enemy_nutrinium': total_enemy_nutrinium,
            'avg_enemy_health': avg_enemy_health,
            'enemy_details': [
                {
                    'name': e.get('name', f'E{i+1}'),  # Include ship name
                    'ai_type': e.get('ai_type', OpponentAIType.BOT_V2),  # Include AI type
                    'model_path': e.get('model_path'),  # Include model path for MODEL type
                    'destroyed': e['destroyed'],
                    'health': e['health'],
                    'credits': e['credits'],
                    'nutrinium': e['nutrinium'],
                    'energy': e.get('energy', 0),
                    'kills': enemy_kills_by_index.get(i, 0),
                    'abilities': dict(e.get('abilities', {}))
                }
                for i, e in enumerate(env.opponent_ships)
            ]
        }

    @staticmethod
    def _format_attack_details(payload: dict) -> str:
        """Format attack payload into a readable combat summary.

        Example output:
          -> E5: dmg 2 (hull 2) | HP 98->96 | ATK[pwr=0 acc=0 nrg=50->49] | DEF[shld=DOWN str=0 evd=0 nrg=60->60]
          -> E3: dmg 3 (hull 3) | HP 80->0 | ... | DESTROYED! wreckage 7 nutr
        """
        if not payload or not isinstance(payload, dict):
            return str(payload) if payload else ''

        target = payload.get('target', '?')
        damage = payload.get('damage', 0)
        destroyed = payload.get('destroyed', False)

        # Combat calculation details
        health_dmg = payload.get('health_dmg', '?')

        # Attacker stats
        atk_energy = payload.get('atk_energy', '?')
        atk_power = payload.get('atk_power', 0)
        atk_accuracy = payload.get('atk_accuracy', 0)

        # Defender stats
        def_health = payload.get('def_health', '?')
        def_shield_state = payload.get('def_shield_state', 'DOWN')
        def_shield_str = payload.get('def_shield_str', 0)
        def_evade = payload.get('def_evade', 0)
        def_energy = payload.get('def_energy', '?')

        parts = []
        parts.append(f"-> {target}: dmg {damage} (hull {health_dmg})")
        parts.append(f"HP {def_health}")
        parts.append(f"ATK[pwr={atk_power} acc={atk_accuracy} nrg={atk_energy}]")
        parts.append(f"DEF[shld={def_shield_state} str={def_shield_str} evd={def_evade} nrg={def_energy}]")

        if destroyed:
            wreckage_nutr = payload.get('wreckage_nutrinium', 0)
            parts.append(f"DESTROYED! wreckage {wreckage_nutr} nutr")

        return ' | '.join(parts)

    @staticmethod
    def _format_mine_details(payload: dict) -> str:
        """Format mining payload into a readable summary.

        Example output (success):
          A(3,2) nutr 20/30 density=0.6667 chance=100.0% | payout 8 -> ship_nutr=15 | mass 30->22 nutr 20->12 | SKILL[acc=2 yield=1 cost=2] nrg=50->45
        Example output (failure):
          A(3,2) nutr 20/30 density=0.6667 chance=50.0% | MISS payout 0 | mass 30->29 nutr 20->20 | SKILL[acc=0 yield=1 cost=2] nrg=50->45
        """
        if not payload or not isinstance(payload, dict):
            return str(payload) if payload else ''

        ax = payload.get('asteroid_x', '?')
        ay = payload.get('asteroid_y', '?')
        ast_mass = payload.get('ast_mass', '?')
        ast_nutr = payload.get('ast_nutr', '?')
        density = payload.get('ast_density', '?')
        chance = payload.get('success_chance', '?')

        payout = payload.get('payout', 0)
        ast_mass_after = payload.get('ast_mass_after', '?')
        ast_nutr_after = payload.get('ast_nutr_after', '?')

        mine_acc = payload.get('mine_accuracy', 0)
        mine_yield = payload.get('mine_yield', 1)
        mine_cost = payload.get('mine_cost_skill', 2)
        energy = payload.get('energy', '?')

        parts = []
        parts.append(f"A({ax},{ay}) nutr {ast_nutr}/{ast_mass} density={density} chance={chance}%")

        if payout > 0:
            ship_nutr = payload.get('ship_nutr', '?')
            parts.append(f"payout (payout) -> ship_nutr={ship_nutr}")
        else:
            parts.append(f"MISS payout 0")

        parts.append(f"mass {ast_mass}->{ast_mass_after} nutr {ast_nutr}->{ast_nutr_after}")
        parts.append(f"SKILL[acc={mine_acc} yield={mine_yield} cost={mine_cost}] nrg={energy}")

        return ' | '.join(parts)

def main():
    """Command-line interface for the simulator."""
    import argparse

    # Set process title for Task Manager visibility
    if SETPROCTITLE_AVAILABLE:
        setproctitle("PnP Playing")
    # Set console window title (more visible on Windows)
    set_console_title("PnP Playing")

    parser = argparse.ArgumentParser(description='Simulate Prospectors n Pirates with trained model')
    parser.add_argument('--model-path', type=str, default='models/ppo_pnp_model',
                       help='Path to trained model. Optional syntax: MODEL_PATH::SPEC_NAME')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'DQN', 'A2C'],
                       help='Algorithm used for the trained model')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--min-opponents', type=int, default=1,
                       help='Minimum number of opponents')
    parser.add_argument('--max-opponents', type=int, default=5,
                       help='Maximum number of opponents')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--render-interval', type=int, default=20,
                       help='Steps between renders')
    parser.add_argument('--pause', action='store_true',
                       help='Pause after each render step (press SPACE to continue)')
    parser.add_argument('--map-width', type=int, default=10,
                       help='Map width')
    parser.add_argument('--map-height', type=int, default=10,
                       help='Map height')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--predefined-asteroids', action='store_true',
                       help='Use predefined asteroids from config file')
    parser.add_argument('--asteroid-config', type=str, default='asteroids_with_trading_posts.config',
                       help='Path to asteroid configuration file')
    parser.add_argument('--predefined-start', action='store_true',
                       help='Use predefined starting positions from config file')
    parser.add_argument('--start-position-config', type=str, default='start_positions.config',
                       help='Path to starting position configuration file')
    parser.add_argument('--cell-width', type=int, default=None,
                       help='Cell width for rendering (default: None)')
    parser.add_argument('--minimap', action='store_true',
                       help='Enable minimap rendering')
    parser.add_argument('--minimap-radius', type=int, default=3,
                       help='Minimap radius (default: 3)')
    parser.add_argument('--print-each-step', action='store_true',
                       help='Print detailed info for each step during simulation')
    parser.add_argument('--opponents', type=str, default=None,
                       help='Comma-separated list of exact opponents (e.g. BOT_V2,BOT_V3,BOT_V4,BOT_V5,BOT_V6,BOT_V7.models/ppo_pnp_model_v29). Overrides --min-opponents and --max-opponents')

    args = parser.parse_args()

    try:
        # Create simulator
        simulator = GameSimulator(
            model_path=args.model_path,
            algorithm=args.algorithm,
            map_width=args.map_width,
            map_height=args.map_height,
            max_steps=args.max_steps,
            use_predefined_asteroids=args.predefined_asteroids,
            asteroid_config_path=args.asteroid_config,
            use_predefined_start=args.predefined_start,
            start_position_config_path=args.start_position_config
        )

        # Parse forced opponents list (supports optional NAME[N] repeat counts)
        forced_opponent_types = None
        if args.opponents:
            forced_opponent_types = _expand_opponents_with_counts(args.opponents)

        # Run simulation
        simulator.run_simulation(
            num_episodes=args.episodes,
            min_opponents=args.min_opponents,
            max_opponents=args.max_opponents,
            render=not args.no_render,
            render_interval=args.render_interval,
            pause_each_step=args.pause,
            cell_width=args.cell_width,
            minimap=args.minimap,
            minimap_radius=args.minimap_radius,
            print_each_step=args.print_each_step,
            forced_opponent_types=forced_opponent_types
        )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
