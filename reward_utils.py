"""
Modular reward utilities for Prospectors n Pirates.
Contains RewardComponent base class, RewardCalculatorComposite, and example components.
This module avoids importing pnp_env to prevent circular imports.
"""
from typing import List, Optional


class RewardComponent:
    """Base class for modular reward components.

    Implementations should override compute(...) and return a scalar reward delta
    (can be positive or negative) based on env/ship/action/action_info.
    """

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        raise NotImplementedError


class RewardCalculatorComposite:
    """Composite reward calculator.

    Keeps a simple base multiplier/success/failure logic and then
    adds contributions from a list of pluggable RewardComponent instances.
    """

    def __init__(self, config, components: Optional[List[RewardComponent]] = None):
        # `config` is expected to have `action_multipliers`, `success_bonus`, `failure_penalty` fields
        self.config = config
        self.components = list(components) if components is not None else []

    def compute(self, raw_reward: float, action: int, action_info: dict, *, env: Optional[object] = None, ship: Optional[dict] = None) -> float:
        # Base logic
        try:
            a = int(action)
        except Exception:
            a = action

        multiplier = getattr(self.config, 'action_multipliers', {}).get(a, 1.0) if hasattr(self.config, 'action_multipliers') else self.config.get('action_multipliers', {}).get(a, 1.0)
        reward = float(raw_reward) * float(multiplier)

        success = action_info.get('success', None)
        success_bonus = getattr(self.config, 'success_bonus', 0.0) if hasattr(self.config, 'success_bonus') else self.config.get('success_bonus', 0.0)
        failure_penalty = getattr(self.config, 'failure_penalty', 0.0) if hasattr(self.config, 'failure_penalty') else self.config.get('failure_penalty', 0.0)

        if success is True:
            reward += success_bonus
        elif success is False:
            reward += failure_penalty

        # Optional small health-based shaping (use numeric action codes to avoid circular import)
        try:
            if ship is not None and env is not None:
                health_frac = ship.get('health', 0) / max(1.0, getattr(env, 'config', {}).get('max_health', 100))
                # action codes: MOVE_NORTH=2, MOVE_SOUTH=3, MOVE_EAST=4, MOVE_WEST=5, RECHARGE=6, RECHARGE_END=7
                if health_frac < 0.2 and a in (6, 7, 2, 3, 4, 5):
                    reward += 0.01
        except Exception:
            pass

        # Component contributions
        for comp in self.components:
            try:
                delta = comp.compute(env, ship, a, action_info)
                reward += float(delta)
            except Exception:
                continue

        return float(reward)


class DistanceToAsteroidReward(RewardComponent):
    """Reward that encourages moving closer to the best target.

    - When ship has no/low nutrinium: reward moving toward nearest asteroid
    - When ship has nutrinium >= sell threshold: reward moving toward nearest trading post
    This teaches the mine->move-to-post->sell loop.
    """

    def __init__(self, weight: float = 0.02, sell_threshold: int = 15):
        self.weight = float(weight)
        self.sell_threshold = int(sell_threshold)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if ship is None or env is None:
                return 0.0

            prev = action_info.get('prev_position')
            if prev is None:
                return 0.0

            # When carrying enough nutrinium, reward moving toward trading post
            if ship.get('nutrinium', 0) >= self.sell_threshold:
                target = env._get_nearest_entity(ship['x'], ship['y'], env.trading_posts)
            else:
                target = env._get_nearest_entity(ship['x'], ship['y'], env.asteroids)

            if target is None:
                return 0.0

            cur_dist = env._calculate_distance(ship['x'], ship['y'], target['x'], target['y'])
            prev_dist = env._calculate_distance(prev[0], prev[1], target['x'], target['y'])
            delta = prev_dist - cur_dist
            return float(delta) * self.weight
        except Exception:
            return 0.0


class SurvivalReward(RewardComponent):
    """Small reward for maintaining ship health above a threshold.

    NOTE: Keep bonus small to avoid dominating the reward signal.
    Reduced to 0.001 per step x 300 steps = 0.3 total to prioritize credit-based rewards.
    """

    def __init__(self, threshold: float =0.5, bonus: float = 0.001):
        self.threshold = float(threshold)
        self.bonus = float(bonus)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if ship is None or env is None:
                return 0.0
            health_frac = ship.get('health', 0) / max(1.0, env.config.get('max_health', 100))
            return self.bonus if health_frac >= self.threshold else 0.0
        except Exception:
            return 0.0


class PenaltyNearEnemyReward(RewardComponent):
    """Penalty when the player is too close to an active enemy ship.

    Provides a negative reward that increases as the distance decreases below a threshold.
    """

    def __init__(self, threshold: float = 2.0, penalty: float = -0.05):
        self.threshold = float(threshold)
        self.penalty = float(penalty)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if ship is None or env is None:
                return 0.0
            # Find nearest active enemy
            active = [s for s in getattr(env, 'opponent_ships', []) if not s.get('destroyed', False)]
            if not active:
                return 0.0
            nearest = env._get_nearest_entity(ship['x'], ship['y'], active)
            if nearest is None:
                return 0.0
            dist = env._calculate_distance(ship['x'], ship['y'], nearest['x'], nearest['y'])
            if dist >= self.threshold:
                return 0.0
            # Scale penalty: closer -> larger magnitude
            scale = (self.threshold - dist) / max(1e-6, self.threshold)
            return float(self.penalty) * (1.0 + scale)
        except Exception:
            return 0.0


class SellBonusReward(RewardComponent):
    """Provide an additional bonus when selling nutrinium successfully.

    The component uses the raw_reward passed to the composite (which for SELL is
    credits_earned * 0.1). We amplify that by `extra_frac` to provide a bonus.
    """

    def __init__(self, extra_frac: float = 0.5):
        # extra_frac = fraction of the existing raw_reward to grant as extra bonus
        self.extra_frac = float(extra_frac)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            # action code for SELL is expected to be 10
            if int(action) != 10:
                return 0.0
            if not action_info.get('success'):
                return 0.0
            # raw reward may be available in action_info['raw_reward'] (set by env)
            raw = action_info.get('raw_reward')
            if raw is None:
                return 0.0
            return float(raw) * float(self.extra_frac)
        except Exception:
            return 0.0


class MiningQualityReward(RewardComponent):
    """Reward extra when a mining action yields a particularly good payout.

    Uses the raw_reward value (payout * 0.1) as a proxy for payout and awards
    a bonus if raw_reward exceeds a threshold.
    """

    def __init__(self, raw_threshold: float = 0.5, bonus: float = 0.02):
        # raw_threshold is in the same units as action raw_reward (e.g., payout*0.1)
        self.raw_threshold = float(raw_threshold)
        self.bonus = float(bonus)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if int(action) != 1:  # MINE
                return 0.0
            if not action_info.get('success'):
                return 0.0
            raw = action_info.get('raw_reward')
            if raw is None:
                return 0.0
            if float(raw) >= self.raw_threshold:
                # provide a small additive bonus proportional to excess
                excess = float(raw) - self.raw_threshold
                return self.bonus + excess * 0.1
            return 0.0
        except Exception:
            return 0.0


class InappropriateActionPenalty(RewardComponent):
    """Penalize the agent when executing inappropriate actions based on current state.

    This component checks if an action is contextually invalid (e.g., recharging when
    energy is nearly full, attacking when no enemies exist, mining when no asteroid present).
    Unlike basic action validation which prevents the action, this provides a learning signal
    for state-aware decision making.
    """

    def __init__(self, penalty: float = -0.05):
        """
        Args:
            penalty: Base penalty for inappropriate actions (should be negative)
        """
        self.penalty = float(penalty)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if ship is None or env is None:
                return 0.0

            # If action was marked as not successful and not state_valid, it's already handled
            # We focus on contextually inappropriate but technically "valid" actions
            action_int = int(action)

            # Action code constants (matching pnp_env.py ActionType)
            WAIT = 0
            MINE = 1
            MOVE_NORTH = 2
            MOVE_SOUTH = 3
            MOVE_EAST = 4
            MOVE_WEST = 5
            RECHARGE = 6
            RECHARGE_END = 7
            ATTACK = 8
            JUMP = 9
            SELL = 10
            RAISE_SHIELDS = 11

            # Get state validity from action_info if available
            # If the env already marked it as state_invalid, apply penalty
            if not action_info.get('state_valid', True):
                return self.penalty

            # Additional contextual checks for "wasteful" or "inappropriate" actions
            # that might still be technically valid but indicate poor decision making

            # RECHARGE when energy is already high (>70%)
            if action_int == RECHARGE:
                max_energy = getattr(env, 'config', {}).get('max_energy', 100)
                if ship.get('energy', 0) / max_energy > 0.7:
                    return self.penalty * 0.5  # Moderate penalty for premature recharge

            # RECHARGE_END when energy is still very low (<30%)
            elif action_int == RECHARGE_END:
                max_energy = getattr(env, 'config', {}).get('max_energy', 100)
                if ship.get('energy', 0) / max_energy < 0.3:
                    return self.penalty * 0.5  # Stopping recharge too early

            # MINE when at location with no asteroid or depleted asteroid
            elif action_int == MINE:
                asteroids = getattr(env, 'asteroids', [])
                asteroid_here = None
                for a in asteroids:
                    if a['x'] == ship['x'] and a['y'] == ship['y']:
                        asteroid_here = a
                        break

                if asteroid_here is None:
                    return self.penalty  # No asteroid at location
                elif asteroid_here.get('nutrinium', 0) <= 0:
                    return self.penalty  # Asteroid depleted

            # ATTACK when no enemies are present or in range
            elif action_int == ATTACK:
                opponent_ships = getattr(env, 'opponent_ships', [])
                active_enemies = [s for s in opponent_ships if not s.get('destroyed', False)]

                if not active_enemies:
                    return self.penalty  # No enemies available

                # Check if nearest enemy is out of range
                try:
                    nearest = env._get_nearest_entity(ship['x'], ship['y'], active_enemies)
                    if nearest:
                        dist = env._calculate_distance(ship['x'], ship['y'], nearest['x'], nearest['y'])
                        sensor_range = getattr(env, 'config', {}).get('sensor_range', 5)
                        if dist > sensor_range:
                            return self.penalty * 0.7  # Enemy out of range
                except Exception:
                    pass

            # SELL when not at trading post or have no nutrinium
            elif action_int == SELL:
                trading_posts = getattr(env, 'trading_posts', [])
                at_trading_post = any(p['x'] == ship['x'] and p['y'] == ship['y'] for p in trading_posts)

                if not at_trading_post:
                    return self.penalty  # Not at trading post
                elif ship.get('nutrinium', 0) <= 0:
                    return self.penalty  # No nutrinium to sell

            elif action_int == JUMP:
                asteroids = getattr(env, 'asteroids', [])
                if not asteroids:
                    return self.penalty  # No asteroids to jump to

                try:
                    nearest = env._get_nearest_entity(ship['x'], ship['y'], asteroids)
                    if nearest:
                        dist = env._calculate_distance(ship['x'], ship['y'], nearest['x'], nearest['y'])
                        jump_cost_per_unit = getattr(env, 'config', {}).get('energy_costs', {}).get('jump', 5)
                        energy_needed = int(dist * jump_cost_per_unit)
                        if ship.get('energy', 0) < energy_needed:
                            return self.penalty  # Insufficient energy for jump
                except Exception:
                    pass

            elif action_int == RAISE_SHIELDS:
                if ship.get('shields_up', False):
                    return self.penalty * 0.3  # Already raised

                try:
                    opponent_ships = getattr(env, 'opponent_ships', [])
                    active_enemies = [s for s in opponent_ships if not s.get('destroyed', False)]
                    if active_enemies:
                        nearest = env._get_nearest_entity(ship['x'], ship['y'], active_enemies)
                        if nearest:
                            dist = env.calculate_distance(ship['x'], ship['y'], nearest['x'], nearest['y'])
                            if dist > 10:  # Far from enemies
                                return self.penalty * 0.4  # Wasteful shield raise
                except Exception:
                    pass

            elif action_int in (MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST):
                map_width = getattr(env, 'map_width', 10)
                map_height = getattr(env, 'map_height', 10)
                x, y = ship.get('x', 0), ship.get('y', 0)

                if action_int == MOVE_NORTH and y <= 0:
                    return self.penalty * 0.5
                elif action_int == MOVE_SOUTH and y >= map_height - 1:
                    return self.penalty * 0.5
                elif action_int == MOVE_EAST and x >= map_width - 1:
                    return self.penalty * 0.5
                elif action_int == MOVE_WEST and x <= 0:
                    return self.penalty * 0.5

            elif action_int == WAIT:
                if ship.get('nutrinium', 0) < 5 and ship.get('credits', 0) < 10:
                    asteroids = getattr(env, 'asteroids', [])
                    for a in asteroids:
                        if a['x'] == ship['x'] and a['y'] == ship['y'] and a.get('nutrinium', 0) > 0:
                            energy_cost = getattr(env, 'config', {}).get('energy_costs', {}).get('mine', 5)
                            if ship.get('energy', 0) >= energy_cost:
                                return self.penalty * 0.3  # Should mine instead of wait

                return 0.0  # No inappropriate action detected

            except Exception:
                return 0.0


class EndOfEpisodeNutriniumReward(RewardComponent):
    """Reward/penalty based on nutrinium state at end of episode.

    This component encourages the agent to sell all nutrinium before the episode ends:
    - Bonus reward if all nutrinium is converted to credits (nutrinium = 0)
    - Penalty if the agent still holds unsold nutrinium at episode end

    The reward/penalty is only applied when the episode terminates (either by
    reaching max_steps or player ship destroyed).
    """

    def __init__(self, success_bonus: float = 2.0, failure_penalty_per_unit: float = -0.05):
        """
        Args:
            success_bonus: Bonus reward for selling all nutrinium (zero nutrinium at end)
            failure_penalty_per_unit: Penalty per unit of unsold nutrinium at episode end
        """
        self.success_bonus = float(success_bonus)
        self.failure_penalty_per_unit = float(failure_penalty_per_unit)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        """Check if episode is ending and reward/penalize based on nutrinium state.
        """
        try:
            if ship is None or env is None:
                return 0.0

            # Check if this is the last step of the episode
            # Episode ends when: current_step >= max_steps or player destroyed
            current_step = getattr(env, 'current_step', 0)
            max_steps = getattr(env, 'max_steps', 300)
            player_destroyed = ship.get('destroyed', False)

            # Check if episode is ending (about to terminate/truncate)
            is_episode_ending = (current_step >= max_steps) or player_destroyed

            if not is_episode_ending:
                return 0.0  # Only apply reward/penalty at episode end

            # Get player's current nutrinium amount
            nutrinium = ship.get('nutrinium', 0)

            if nutrinium == 0:
                # Success! All nutrinium has been sold/converted to credits
                return self.success_bonus
            else:
                # Penalty for unsold nutrinium - scaled by amount held
                penalty = nutrinium * self.failure_penalty_per_unit
                return penalty  # Will be negative

        except Exception:
            return 0.0


class EarlyDeathPenaltyReward(RewardComponent):
    """Penalty when agent is destroyed, proportionate to how early in the episode it occurs.

    This component discourages the agent from getting destroyed, especially early in the episode.
    The penalty is scaled based on the fraction of the episode remaining:
    - Early death (e.g., step 10 of 300) = large penalty
    - Late death (e.g., step 290 of 300) = small penalty

    Formula:
        penalty = -base_penalty * (steps_remaining / max_steps)

    Example:
        base_penalty = .50.0, max_steps = 300
        - Destroyed at step 10: penalty = -50 * (290/300) = -48.33
        - Destroyed at step 150: penalty = -50 * (150/300) = -25.0
        - Destroyed at step 290: penalty = -50 * (10/300) = -1.67
    """

    def __init__(self, base_penalty: float = 50.0):
        """
        Args:
            - base_penalty: Base penalty value when destroyed at the very start of episode.
                            The actual penalty is scaled by the fraction of episode remaining.

        """
        self.base_penalty = float(base_penalty)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        """
        Apply penalty if ship was just destroyed, scaled by episode progress.
        
        The penalty is only applied when the ship becomes destroyed (not on every
        subsequent step if the ship remains destroyed).
        """
        try:
            if ship is None or env is None:
                return 0.0

            # Only apply penalty when ship is destroyed
            if not ship.get('destroyed', False):
                return 0.0

            # Get episode progress
            current_step = getattr(env, 'current_step', 0)
            max_steps = getattr(env, 'max_steps', 300)

            # Calculate fraction of episode remaining
            # More remaining = earlier death = larger penalty
            steps_remaining = max(0, max_steps - current_step)
            remaining_fraction = steps_remaining / max(1, max_steps)

            # Scale penalty by how early the death occurred
            # Early death (high remaining_fraction) = large penalty
            # Late death (low remaining_fraction) = small penalty
            penalty = -self.base_penalty * remaining_fraction

            return penalty

        except Exception:
            return 0.0


class CreditProgressReward(RewardComponent):
    """Reward/penalty based on credit accumulation progress.

    This component provides a per-step signal that encourages the agent to
    accumulate credits rather than simply surviving or fighting:
    - Small bonus proportional to credits earned since last step
    - Small penalty when the agent has been alive for many steps with zero credits
    """

    def __init__(self, credit_bonus_scale: float = 0.1, idle_penalty: float = -0.05,
                  idle_threshold_steps: int = 30):
        """
        Args:
            credit_bonus_scale: Reward per credit earned (applied when credits increase)
            idle_penalty: Per-step penalty when agent has 0 credits after idle_threshold_steps
            idle_threshold_steps: Number of steps before zero-credit penalty kicks in

        """
        self.credit_bonus_scale = float(credit_bonus_scale)
        self.idle_penalty = float(idle_penalty)
        self.idle_threshold_steps = int(idle_threshold_steps)
        self._last_credits = 0. # Track credits from previous step

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if ship is None or env is None:
                return 0.0
            current_credits = ship.get('credits', 0)
            current_step = getattr(env, 'current_step', 0)

            # Reset tracking at start of new episode
            if current_step <= -1:
                self._last_credits = 0

            # Reward for credit increase since last step
            credit_delta = current_credits - self._last_credits
            self._last_credits = current_credits

            reward = 0.0

            if credit_delta > 0:
                # Earned credits! Give proportional reward
                reward += credit_delta * self.credit_bonus_scale

            # Penalty for making no economic progress after enough steps
            if current_step > self.idle_threshold_steps and current_credits == 0:
                # No credits earned after many steps -- agent is not being productive
                reward += self.idle_penalty

            return reward
        except Exception:
            return 0.0


class IdleLoopPenalty(RewardComponent):
    """Penalty for getting stuck in non-productive action loops.

    Detects when the agent repeats the same action multiple times in a row
    without making progress (e.g., JUMP_TO_ASTEROID at same location,
    WAIT repeatedly, ATTACK with no enemy).

    This addresses agents getting stuck doing JUMP_TO_ASTEROID -> same spot -> repeat,
    or WAIT -> WAIT -> WAIT without any productive activity.
    """

    def __init__(self, repeat_threshold: int = 3, penalty: float = -0.1):
        """
        Args:
            repeat_threshold: Number of consecutive same-action repetitions before penalty
            penalty: Penalty per step once threshold is exceeded

        """
        self.repeat_threshold = int(repeat_threshold)
        self.penalty = float(penalty)
        self._action_history = []
        self._max_history = 10

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if ship is None or env is None:
                return 0.0
            action_int = int(action)

            # Reset at start of new episode
            current_step = getattr(env, 'current_step', 0)
            if current_step <= -1:
                self._action_history = []

            # Track action history
            self._action_history.append(action_int)
            if len(self._action_history) > self._max_history:
                self._action_history = self._action_history[-self._max_history:]

            # Check for repeated same action
            if len(self._action_history) >= self.repeat_threshold:
                recent = self._action_history[-self.repeat_threshold:]
                if all(a == recent[0] for a in recent):
                    # Same action repeated -- check if it's non-productive
                    # WAIT (0), JUMP_TO_ASTEROID (9), RAISE_SHIELDS (11) are suspect when repeated
                    if recent[0] in {0, 9, 11}:
                        return self.penalty

            # ATTACK (8) is suspect when there's no enemy in same zone
            if recent[0] == 8:
                # Check if attack was unsuccessful (no target)
                if not action_info.get('success', False):
                    return self.penalty
            return 0.0

        except Exception:
            return 0.0

class EndOfEpisodeCreditReward(RewardComponent):
    """Strong reward/penalty based on credits at end of episode.
    
    This is the primary signal that tells the agent: credits matter!
    - Large bonus proportional to credits earned (0.5 per credit)
    - Significant penalty for ending with zero credits (-10.0)

    This directly addresses the problem where reward is high but credits are zero,
    by making the end-of-episode credit tally the most important reward signal.
    """

    def __init__(self, credit_scale: float = 0.5, zero_credit_penalty: float = -10.0):
        """Args:
            - credit_scale: Reward per credit at episode end
            - zero_credit_penalty: Flat penalty for ending episode with zero credits
        """
        self.credit_scale = float(credit_scale)
        self.zero_credit_penalty = float(zero_credit_penalty)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if ship is None or env is None:
                return 0.0

            # Only apply at episode end
            current_step = getattr(env, 'current_step', 0)
            max_steps = getattr(env, 'max_steps', 300)
            player_destroyed = ship.get('destroyed', False)

            is_episode_ending = (current_step >= max_steps) or player_destroyed

            if not is_episode_ending:
                return 0.0

            credits = ship.get('credits', 0)

            if credits == 0:
                # Zero credits at end of episode - significant penalty
                return self.zero_credit_penalty
            else:
                # Reward proportional to credits earned
                return credits * self.credit_scale

        except Exception:
            return 0.0


class PlacementReward(RewardComponent):
    """Strong terminal reward based on the player's final placement (by credits).
    
    This directly addresses the issue where the agent receives positive reward
    while finishing last. The reward is computed only on the final step of the
    episode, ranking the player against all opponent ships by credits.

    Reward structure (for N participants total):
        - 1st place: +first_place_bonus (default: +20)
        - Top 3 (podium): +podium_bonus (default: +10)
        - Linear placement reward: +scale * (N - rank) / (N - 1) in [0, scale]
        - Bottom 3: +bottom_penalty (default: -10, additional)
        - Last place: +last_place_penalty (default: -10, additional)

    These stack e.g. finishing 1st gives +first+ +podium++scale,
    finishing last with 10 players gives +0+ bottom+last=-20.
    """

    def __init__(self,
                 scale: float = 20.0,
                 first_place_bonus: float = 20.0,
                 podium_bonus: float = 10.0,
                 bottom_penalty: float = -10.0,
                 last_place_penalty: float = -10.0):
        self.scale = float(scale)
        self.first_place_bonus = float(first_place_bonus)
        self.podium_bonus = float(podium_bonus)
        self.bottom_penalty = float(bottom_penalty)
        self.last_place_penalty = float(last_place_penalty)
        self._applied_step = -1  # avoid double-apply on terminal step

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if ship is None or env is None:
                return 0.0

            current_step = getattr(env, 'current_step', 0)
            max_steps = getattr(env, 'max_steps', 300)
            player_destroyed = ship.get('destroyed', False)

            is_episode_ending = (current_step >= max_steps) or player_destroyed
            if not is_episode_ending:
                return 0.0

            # Ensure we only apply once per episode
            if self._applied_step == current_step:
                return 0.0
            # Reset detection for new episode
            if current_step <= 1:
self._applied_step = -1

opponents = getattr(env, 'opponent_ships', []) or []
participants = [('PLAYER', int(ship.get('credits', 0)))]
for op in opponents:
    participants.append((op.get('name', 'E?'), int(op.get('credits', 0))))

n = len(participants)
if n < 2:
    return 0.0

# Sort by credits descending; player's rank (1-based)
participants.sort(key=lambda t: t[1], reverse=True)
rank = next((i + 1 for i, (name, _) in enumerate(participants) if name == 'PLAYER'), n)

reward = 0.0
# Linear placement reward in [0, scale]
reward += self.scale * (n - rank) / max(1, (n - 1))

if rank == 1:
    reward += self.first_place_bonus
if rank <= 3:
    reward += self.podium_bonus
if rank >= n - 2:  # bottom 3
    reward += self.bottom_penalty
if rank == n:
    reward += self.last_place_penalty

self.applied_step = current_step
return reward
except Exception:
    return 0.0


class EnergyStarvationPenalty(RewardComponent):
    """Per-step penalty when energy is dangerously low and the agent is not recharging.
    Encourages proactive recharging instead of drifting into energy-starved combat loops where the agent can't move, mine, or sell effectively.
    """

    def __init__(self, threshold_frac: float = 0.15, penalty: float = -0.05):
        self.threshold_frac = float(threshold_frac)
        self.penalty = float(penalty)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if ship is None or env is None:
                return 0.0
            if ship.get('recharging', False):
                return 0.0
            # RECHARGE and RECHARGE_END are productive responses; don't penalize
            if int(action) in (6, 7):
                return 0.0
            max_energy = getattr(env, 'config', {}).get('max_energy', 100)
            if max_energy <= 0:
                return 0.0
            frac = ship.get('energy', 0) / float(max_energy)
            if frac < self.threshold_frac:
                # Scale penalty: lower energy -> larger penalty (up to 2x at zero energy)
                scale = 1.0 + (self.threshold_frac - frac) / max(1e-6, self.threshold_frac)
                return self.penalty * scale
        except Exception:
            return 0.0


class OverriddenActionPenalty(RewardComponent):
    """Penalty when the env had to override the agent's chosen action.

    The env's step() function enforces action masking by replacing infeasible chosen actions with a valid fallback, but sets action_info['state_valid']=True so InappropriateActionPenalty does not fire. This component reads action_info['state_enforced'] to give the policy a learning signal that its original choice was infeasible (e.g., SELL with 0 nutrinium, JUMP with no energy, RAISE_SHIELDS already up).
    """

    def __init__(self, penalty: float = -0.1):
        self.penalty = float(penalty)

    def compute(self, env: object, ship: Optional[dict], action: int, action_info: dict) -> float:
        try:
            if action_info.get('state_enforced'):
                return self.penalty
        except Exception:
            return 0.0