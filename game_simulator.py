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
import gymnasium as gym

try:
    from stable_baselines3 import PPO, DQN, A2C
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    from sb3_contrib import MaskablePPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    MaskablePPO = None

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


class ActionMaskTracker(gym.Wrapper):
    """Exposes ``action_masks()`` so MaskablePPO can read the mask at every step.

    When sb3-contrib's MaskablePPO is available the environment must expose a
    callable ``action_masks()`` method.  The raw ``ProspectorsPiratesEnv``
    stores the mask inside the observation dict, so this wrapper:
    1. Caches the action mask and exposes it via action_masks()
    2. Strips the action_mask from the returned observation dict
    3. Flattens the observation to a plain Box for MaskablePPO compatibility
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._action_mask = np.ones(env.action_space.n, dtype=bool)
        
        # Change observation space from Dict to just the flat Box observation
        if isinstance(env.observation_space, gym.spaces.Dict) and 'observation' in env.observation_space.spaces:
            self.observation_space = env.observation_space.spaces['observation']

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._extract_and_cache_mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._extract_and_cache_mask(obs)
        return obs, reward, terminated, truncated, info

    def _extract_and_cache_mask(self, obs):
        """Extract action mask from obs dict and return only the flat observation."""
        if isinstance(obs, dict):
            if 'action_mask' in obs:
                self._action_mask = np.array(obs['action_mask'], dtype=bool)
            if 'observation' in obs:
                return obs['observation']
        return obs

    def action_masks(self) -> np.ndarray:
        """Return the latest action mask - called by MaskablePPO at every step."""
        return self._action_mask
    
    def __getattr__(self, name):
        """Forward attribute lookups to the wrapped environment."""
        return getattr(self.env, name)


class ActionSpaceCompatibilityWrapper:
    """Wrapper to make old models (15 actions) compatible with current environment (14 actions).

    Old models may have been trained with LOWER_SHIELDS (action 12) which no longer exists.
    Actions 0-11 map directly. Old action 12 (LOWER_SHIELDS) maps to WAIT.
    Old action 13 (JUMP_TO_TRADING_POST) maps to new 12. Old action 14 (RESPAWN) maps to new 13.

    Also handles observation space compatibility: old models expect a flat numpy array
    but the new environment returns a Dict with 'observation' and 'action_mask'.

    Also handles observation size compatibility: old models may expect fewer features
    (e.g. 128 or 192) vs current environment (e.g. 200). Observation is truncated to fit.
    """
    def __init__(self, model, old_action_space_size=15, new_action_space_size=14):
        self.model = model
        self.old_size = old_action_space_size
        self.new_size = new_action_space_size
        # Detect if model expects flat observation (Box) vs Dict
        from gymnasium import spaces
        model_obs = getattr(self.model, 'observation_space', None)
        self._expects_flat_obs = isinstance(model_obs, spaces.Box)
        # Detect if model expects different observation size (Dict with smaller obs vector)
        self._model_obs_size = None
        if isinstance(model_obs, spaces.Dict) and 'observation' in model_obs.spaces:
            self._model_obs_size = model_obs['observation'].shape[0]
        elif isinstance(model_obs, spaces.Box):
            self._model_obs_size = model_obs.shape[0]

    def predict(self, observation, deterministic=False):
        """Predict action using the old model, handling action/observation space differences."""
        # If model expects flat obs but we got a Dict, extract the flat array
        if self._expects_flat_obs and isinstance(observation, dict):
            observation = observation['observation']

        # Truncate observation if model expects fewer features than provided
        if self._model_obs_size is not None:
            import numpy as np
            if isinstance(observation, dict) and 'observation' in observation:
                obs_arr = observation['observation']
                if obs_arr.shape[-1] > self._model_obs_size:
                    observation = dict(observation)
                    observation['observation'] = obs_arr[..., :self._model_obs_size]
            elif hasattr(observation, 'shape') and observation.shape[-1] > self._model_obs_size:
                observation = observation[..., :self._model_obs_size]

        action, state = self.model.predict(observation, deterministic=deterministic)

        # Remap old 15-action model output to current 14-action space:
        # 0-11: same (WAIT through RAISE_SHIELDS)
        # 12 (old LOWER_SHIELDS): map to 0 (WAIT) since action no longer exists
        # 13 (old JUMP_TO_TRADING_POST): map to 12
        # 14 (old RESPAWN): map to 13
        action_int = int(action) if hasattr(action, '__int__') else int(action.item())
        if self.old_size > self.new_size:
            if action_int == 12:
                action_int = 0  # LOWER_SHIELDS -> WAIT
            elif action_int == 13:
                action_int = 12  # JUMP_TO_TRADING_POST
            elif action_int == 14:
                action_int = 13  # RESPAWN
            import numpy as np
            action = np.array(action_int)

        return action, state


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
        "RAISE_SHIELDS", "JUMP_TO_TRADING_POST", "RESPAWN"
    ]

    @staticmethod
    def get_ai_type_name(ai_type):
        """Convert AI type enum to readable string"""
        if ai_type == OpponentAIType.PROSPECTOR:
            return "PROSPECTOR"
        elif ai_type == OpponentAIType.PIRATE:
            return "PIRATE"
        elif ai_type == OpponentAIType.HEURISTIC:
            return "HEURISTIC"
        elif ai_type == OpponentAIType.MODEL:
            return "MODEL"
        else:
            return "UNKNOWN"

    @staticmethod
    def _format_attack_details(payload: dict) -> str:
        """Format attack payload into a readable combat summary.

        Example output:
          -> E5: dmg 2 (base 2, roll 2.4, shields -0) | HP 98->96 | ATK pwr=0 nrg=50->49 | DEF shields=N evd=0 nrg=60->60
          -> E3: dmg 1 (base 2, roll 1.8, shields -1) | HP 80->79 shields=Y | DESTROYED! stole 15 nutr
        """
        if not payload or not isinstance(payload, dict):
            return str(payload) if payload else ''

        target = payload.get('target', '?')
        damage = payload.get('damage', 0)
        destroyed = payload.get('destroyed', False)

        # Combat calculation details
        base_dmg = payload.get('base_dmg', '?')
        dmg_roll = payload.get('dmg_roll', '?')
        shield_absorbed = payload.get('shield_absorbed', 0)

        # Attacker stats
        atk_energy = payload.get('atk_energy', '?')
        atk_power = payload.get('atk_power', 0)
        atk_accuracy = payload.get('atk_accuracy', 0)

        # Defender stats
        def_health = payload.get('def_health', '?')
        def_shields = payload.get('def_shields', False)
        def_shield_str = payload.get('def_shield_str', 0)
        def_evade = payload.get('def_evade', 0)
        def_energy = payload.get('def_energy', '?')

        parts = []
        parts.append(f"-> {target}: dmg {damage} (base {base_dmg}, roll {dmg_roll}, shields -{shield_absorbed})")
        parts.append(f"HP {def_health}")
        parts.append(f"ATK[pwr={atk_power} acc={atk_accuracy} nrg={atk_energy}]")
        shield_flag = 'Y' if def_shields else 'N'
        parts.append(f"DEF[shld={shield_flag} str={def_shield_str} evd={def_evade} nrg={def_energy}]")

        if destroyed:
            nutr_stolen = payload.get('nutrinium_stolen', 0)
            parts.append(f"DESTROYED! stole {nutr_stolen} nutr")

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
            parts.append(f"payout {payout} -> ship_nutr={ship_nutr}")
        else:
            parts.append(f"MISS payout 0")

        parts.append(f"mass {ast_mass}->{ast_mass_after} nutr {ast_nutr}->{ast_nutr_after}")
        parts.append(f"SKILL[acc={mine_acc} yield={mine_yield} cost={mine_cost}] nrg={energy}")

        return ' | '.join(parts)

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
        self.model_path = model_path
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
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            # Create timestamped directory name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(output_base_dir, f"{model_name}_{timestamp}")
            self.logger = SimulationLogger(output_dir, enable_logging=True)

        # Validate model exists
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")

        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Statistics tracking
        self.episode_stats: List[Dict] = []

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
                # Try MaskablePPO first (newer models), fall back to regular PPO
                if SB3_CONTRIB_AVAILABLE:
                    try:
                        temp_model = MaskablePPO.load(self.model_path)
                    except Exception:
                        temp_model = PPO.load(self.model_path)
                else:
                    temp_model = PPO.load(self.model_path)
            elif self.algorithm == "DQN":
                temp_model = DQN.load(self.model_path)
            elif self.algorithm == "A2C":
                temp_model = A2C.load(self.model_path)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            # Check if action spaces match
            model_action_space = temp_model.action_space.n
            env_action_space = env.action_space.n

            needs_action_compat = False
            if model_action_space != env_action_space:
                print(f"\nWARNING: Model action space ({model_action_space}) != Environment action space ({env_action_space})")
                print(f"This model was trained with an older version of the environment.")

                if model_action_space == 15 and env_action_space == 14:
                    print(f"Detected: Old model (15 actions with LOWER_SHIELDS) vs Current environment (14 actions)")
                    print(f"Solution: Using compatibility mode - old LOWER_SHIELDS mapped to WAIT")
                    print(f"Note: For best results, retrain the model with the new action space\n")
                    needs_action_compat = True
                else:
                    raise ValueError(f"Incompatible action spaces: {model_action_space} vs {env_action_space}")

            # Check if model expects flat observation (Box) vs new Dict space
            model_obs_space = temp_model.observation_space
            needs_obs_compat = isinstance(model_obs_space, spaces.Box) and isinstance(env.observation_space, spaces.Dict)

            if needs_obs_compat:
                print(f"  Model expects flat observation (Box), environment provides Dict.")
                print(f"  Using compatibility wrapper to extract flat observation.\n")

            # Check if model expects Dict obs with a different (smaller) observation size
            needs_obs_size_compat = False
            if (isinstance(model_obs_space, spaces.Dict) and
                    isinstance(env.observation_space, spaces.Dict) and
                    'observation' in model_obs_space.spaces and
                    'observation' in env.observation_space.spaces):
                model_obs_size = model_obs_space['observation'].shape[0]
                env_obs_size = env.observation_space['observation'].shape[0]
                if model_obs_size != env_obs_size:
                    print(f"\nWARNING: Model observation size ({model_obs_size}) != Environment observation size ({env_obs_size})")
                    print(f"This model was trained with an older version of the environment.")
                    print(f"Using compatibility mode - observation will be truncated to {model_obs_size} features.\n")
                    needs_obs_size_compat = True

            # Use compatibility wrapper if needed for action, observation type, or obs size
            if needs_action_compat or needs_obs_compat or needs_obs_size_compat:
                return ActionSpaceCompatibilityWrapper(
                    temp_model,
                    model_action_space,
                    env_action_space
                )

            # Fully compatible model - load with env binding
            if self.algorithm == "PPO":
                # Try MaskablePPO first (newer models), fall back to regular PPO
                if SB3_CONTRIB_AVAILABLE:
                    try:
                        return MaskablePPO.load(self.model_path, env=env)
                    except Exception:
                        return PPO.load(self.model_path, env=env)
                else:
                    return PPO.load(self.model_path, env=env)
            elif self.algorithm == "DQN":
                return DQN.load(self.model_path, env=env)
            elif self.algorithm == "A2C":
                return A2C.load(self.model_path, env=env)

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
        
        # Apply ActionMaskTracker wrapper to match training environment
        env = ActionMaskTracker(env)

        # Load model
        model = self.load_model(env)

        # Run episode
        observation, info = env.reset()

        # Print opponent AI types at start
        if print_each_step or pause_each_step:
            print(f"\n{'='*60}")
            print("OPPONENTS:")
            for i, enemy in enumerate(env.opponent_ships):
                e_name = enemy.get('name', f'E{i+1}')
                ai_type = enemy.get('ai_type', 0)
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

        total_reward = 0
        done = False
        step = 0
        control = {'skip': False, 'quit': False}

        # Track action distribution for debugging
        action_counts = {}
        debug_enabled = False  # Set to False to disable debug output

        while not done:
            # Check if player is destroyed - if so, use RESPAWN action
            if env.player_ship.get('destroyed', False):
                action = 13  # RESPAWN action
                original_action = 13
                original_type = int
            else:
                # Model predicts action
                action_raw, _ = model.predict(observation, deterministic=deterministic)

                # DEBUG: Track original action before conversion
                original_action = action_raw
                original_type = type(action_raw)

                # Defensive: convert numpy array/scalar action to int
                if isinstance(action_raw, np.ndarray):
                    try:
                        action = int(action_raw.item())
                    except Exception:
                        action = int(action_raw[0])
                else:
                    try:
                        action = int(action_raw)
                    except Exception:
                        action = action_raw

            # DEBUG: Print conversion details on first step
            if step == 0 and debug_enabled:
                print(f"\n[DEBUG] First action prediction:")
                print(f"  Original: {original_action}, Type: {original_type}")
                print(f"  Converted: {action}, Type: {type(action)}")
                print(f"  Is numpy array: {isinstance(original_action, np.ndarray)}")
                if isinstance(original_action, np.ndarray):
                    print(f"  Array shape: {original_action.shape}")
                    print(f"  Array dtype: {original_action.dtype}")

            # Track action distribution
            action_counts[action] = action_counts.get(action, 0) + 1

            # NOTE: env.step() is now called inside the rendering block below
            # (moved to allow showing state BEFORE action execution)

            # Log/render periodically - BEFORE executing action so we see current state
            # Always log details to file when logger enabled, but only show in console when render=True
            should_log_details = (self.logger is not None) or (render and step % render_interval == 0)
            if should_log_details and step % render_interval == 0:
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

                # Check if the predicted action is valid; if not, show what will actually execute
                is_valid, _reason = env._is_action_valid_for_state(action, p, is_player=True)
                if not is_valid:
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
                    predicted_action_name = f"{enforced} (was {predicted_action_name})"

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
                    widths[7] = max(widths[7], len(ship['action']))

                # Print header
                header_line = '  '.join(h.ljust(w) for h, w in zip(headers, widths))
                sep_line = '  '.join('-' * w for w in widths)
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

                # DEBUG: Verify action value before name lookup
                if debug_enabled and step == 0:
                    print(f"\n[DEBUG] Action name lookup:")
                    print(f"  Predicted action = {action}")
                    print(f"  action < len(ACTION_NAMES) = {action < len(self.ACTION_NAMES)}")
                    print(f"  len(ACTION_NAMES) = {len(self.ACTION_NAMES)}")

                # Pause if requested (before action execution)
                if pause_each_step and not control['skip'] and not control['quit']:
                    rv = self._wait_for_spacebar()
                    if rv == 'skip':
                        control['skip'] = True
                    elif rv == 'quit':
                        control['quit'] = True
                        # Set done to True to break out and finish this episode early
                        done = True

            # 4. EXECUTE THE ACTION
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # 5. SHOW ACTION RESULT (after execution) - TABLE FORMAT FOR ALL SHIPS
            # Always log to file when logger enabled, but only show in console when render=True
            if should_log_details and step % render_interval == 0:
                self._log_episode_detail(f"\n{'=' * 70}", render)
                self._log_episode_detail(f"ACTION RESULTS (Step {step})", render)
                self._log_episode_detail(f"{'=' * 70}", render)

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
                        'reward': '-',  # Don't show enemy rewards
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
                header_line = '  '.join(h.ljust(w) for h, w in zip(headers, widths))
                sep_line = '  '.join('-' * w for w in widths)
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
                    line = '  '.join(row)
                    # Highlight player row
                    if ar['is_player']:
                        self._log_episode_detail(f"? {line}", render)
                    else:
                        self._log_episode_detail(f"  {line}", render)

                # Show total reward for player
                self._log_episode_detail(f"\nPlayer Total Reward: {total_reward:.3f}", render)

                # Show state validation warning if needed
                if not state_valid_flag:
                    self._log_episode_detail(f"WARN Warning: Action state invalid - {info.get('state_invalid_reason','')}", render)

                # DEBUG: Show action distribution periodically
                if debug_enabled and step % (render_interval * 5) == 0:
                    print(f"[DEBUG] Action distribution so far: {action_counts}")


            step += 1

        # DEBUG: Print final action distribution
        if debug_enabled and render:
            print(f"\n[DEBUG] Final action distribution for episode:")
            for act, count in sorted(action_counts.items()):
                act_name = self.ACTION_NAMES[act] if act < len(self.ACTION_NAMES) else f"ACTION_{act}"
                percentage = (count / step) * 100 if step > 0 else 0
                print(f"  {act_name:15} (Action {act}): {count:4} times ({percentage:5.1f}%)")

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

        # Gather statistics
        enemies_alive = sum(1 for e in env.opponent_ships if not e['destroyed'])
        enemies_destroyed = num_opponents - enemies_alive
        total_enemy_credits = sum(e['credits'] for e in env.opponent_ships)
        total_enemy_nutrinium = sum(e['nutrinium'] for e in env.opponent_ships if not e['destroyed'])
        avg_enemy_health = np.mean([e['health'] for e in env.opponent_ships if not e['destroyed']]) if enemies_alive > 0 else 0

        episode_stats = {
            'num_opponents': num_opponents,
            'steps': step,
            'total_reward': total_reward,
            'player_credits': info.get('player_credits', 0),
            'player_nutrinium': info.get('player_nutrinium', 0),
            'player_health': info.get('player_health', 0),
            'player_energy': info.get('player_energy', 0),
            'player_destroyed': info.get('player_destroyed', False),
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
                    'ai_type': e.get('ai_type', 0),  # Include AI type
                    'model_path': e.get('model_path'),  # Include model path for MODEL type
                    'destroyed': e['destroyed'],
                    'health': e['health'],
                    'credits': e['credits'],
                    'nutrinium': e['nutrinium'],
                    'abilities': dict(e.get('abilities', {}))
                }
                for i, e in enumerate(env.opponent_ships)
            ]
        }

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
        #     print(f"    {enemy_name} ({ai_display}): {status}, HP:{enemy['health']}, "
        #           f"Credits:{enemy['credits']}, Nutrinium:{enemy['nutrinium']}")
        #     # Print enemy abilities/skills horizontally
        #     e_abilities = enemy.get('abilities', {}) or {}
        #     if e_abilities:
        #         e_items = ", ".join(f"{k}:{v}" for k, v in sorted(e_abilities.items()))
        #         print(f"      Abilities: {e_items}")

    def calculate_player_placement(self, stats: Dict):
        """Calculate player's placement (1st, 2nd, 3rd) for this episode and update cumulative stats."""
        # Collect all participants with their credits and nutrinium
        participants = []

        # Player
        p_credits = stats.get('player_credits', 0) or 0
        p_nutrinium = stats.get('player_nutrinium', 0) or 0
        participants.append({
            'name': 'PLAYER',
            'credits': p_credits,
            'nutrinium': p_nutrinium,
            'is_player': True
        })

        # Enemies
        for enemy in stats.get('enemy_details', []):
            e_credits = enemy.get('credits', 0) or 0
            e_nutrinium = enemy.get('nutrinium', 0) or 0
            participants.append({
                'name': enemy.get('name', 'ENEMY'),
                'credits': e_credits,
                'nutrinium': e_nutrinium,
                'is_player': False
            })

        # Sort by credits (descending), then by nutrinium (descending)
        participants.sort(key=lambda p: (p['credits'], p['nutrinium']), reverse=True)

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
        rows = []

        # Player row
        p_name = 'PLAYER'
        p_role = 'PLAYER'
        p_credits = int(stats.get('player_credits', 0) or 0)
        p_health = int(stats.get('player_health', 0) or 0)
        p_nutr = int(stats.get('player_nutrinium', 0) or 0)
        p_energy = int(stats.get('player_energy', 0)) if stats.get('player_energy') is not None else ''
        p_status = 'DESTROYED' if stats.get('player_destroyed', False) else 'ALIVE'
        p_abilities = stats.get('player_abilities') or {}
        p_abilities_str = ", ".join(f"{k}:{v}" for k, v in sorted(p_abilities.items())) if p_abilities else ''
        rows.append((p_name, p_role, p_credits, p_health, p_nutr, p_energy, p_status, p_abilities_str))

        # Enemies
        for enemy in stats.get('enemy_details', []):
            e_name = enemy.get('name', 'ENEMY')
            ai_type = enemy.get('ai_type', None)
            e_role = self.get_ai_type_name(ai_type) if ai_type is not None else 'ENEMY'
            if ai_type == OpponentAIType.MODEL:
                model_path = enemy.get('model_path', 'unknown')
                model_name = os.path.basename(model_path) if model_path else 'unknown'
                e_role = f"{e_role}:{model_name}"
            e_credits = int(enemy.get('credits', 0) or 0)
            e_health = int(enemy.get('health', 0) or 0)
            e_nutr = int(enemy.get('nutrinium', 0) or 0)
            e_energy = ''
            e_status = 'DESTROYED' if enemy.get('destroyed', False) else 'ALIVE'
            e_abilities = enemy.get('abilities') or {}
            e_abilities_str = ", ".join(f"{k}:{v}" for k, v in sorted(e_abilities.items())) if e_abilities else ''
            rows.append((e_name, e_role, e_credits, e_health, e_nutr, e_energy, e_status, e_abilities_str))

        # Sort by credits descending
        rows.sort(key=lambda r: r[2], reverse=True)

        # Prepare header and widths
        headers = ["Ship", "Role", "Credits", "Health", "Nutrinium", "Energy", "Status", "Abilities"]
        cols = list(zip(*([[str(h) for h in headers]] + [[str(v) for v in row] for row in rows])))
        widths = [max(len(cell) for cell in col) for col in cols]

        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        sep_line = "  ".join('-' * w for w in widths)
        self._print("\n" + header_line)
        self._print(sep_line)

        for row in rows:
            line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            self._print(line)

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
            episode_rankings.append(('PLAYER', p_credits))

            # Enemies
            for enemy in stats.get('enemy_details', []):
                e_name = enemy.get('name', 'ENEMY')
                e_credits = enemy.get('credits', 0) or 0
                episode_rankings.append((e_name, e_credits))

            # Sort by credits descending to get rankings
            episode_rankings.sort(key=lambda x: x[1], reverse=True)

            # Create placement map: participant_name -> placement (1, 2, 3, ...)
            placements = {name: rank + 1 for rank, (name, _) in enumerate(episode_rankings)}

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
            p['total_enemies_destroyed'] += stats.get('enemies_destroyed', 0)
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
            line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            self._print(line, to_episode=False)
        self._print("=" * 70, to_episode=False)

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
                       help='Path to trained model')
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
                       help='Comma-separated list of exact opponents (e.g. HEURISTIC,PIRATE,PROSPECTOR,models/ppo_pnp_model_v29). Overrides --min-opponents and --max-opponents.')

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

        # Parse forced opponents list
        forced_opponent_types = None
        if args.opponents:
            forced_opponent_types = [s.strip() for s in args.opponents.split(',')]

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
