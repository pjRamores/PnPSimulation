"""Action-masking / validation mixin for :class:`ProspectorsPiratesEnv`.

Owns action normalization (:meth:`_normalize_action`), energy-bin decoding
(:meth:`_energy_from_bin`), the per-state validity rules
(:meth:`_is_action_valid_for_state`) and the boolean mask builder
(:meth:`_get_action_mask`).
"""

from env_common import *

from utils import action_masker


class EnvMaskingMixin:
    """Action normalization and state-aware validity masking."""

    def _normalize_action(self, action) -> Tuple[int, Optional[Tuple[int, int]], Optional[int]]:
        """Normalize scalar / array / dict actions into ``(action_type, target, energy)``.

        Backward compatible: a bare scalar (or length-1 array) yields ``(atype, None, None)``,
        preserving the legacy auto-target / default-energy behaviour relied on by the
        heuristic AIs and scalar-Discrete callers. A structured action (a dict matching the
        env's Dict action space, or a length-4 array ``[atype, tx, ty, energy_bin]``) is
        parsed into an explicit target coordinate and an energy amount (via the energy bin).
        """
        try:
            # Dict action (matches the env's Dict action_space)
            if isinstance(action, dict):
                atype = int(np.asarray(action.get('action_type', 0)).item())
                target = None
                tgt = action.get('target', None)
                if tgt is not None:
                    flat = np.asarray(tgt).flatten()
                    if flat.size >= 2:
                        target = (int(flat[0]), int(flat[1]))
                energy = None
                ebin = action.get('energy', None)
                if ebin is not None:
                    energy = self._energy_from_bin(int(np.asarray(ebin).item()))
                return atype, target, energy
            # Numpy array (length-4 structured, else scalar-like)
            if isinstance(action, np.ndarray):
                flat = action.flatten()
                if flat.size >= 4:
                    return int(flat[0]), (int(flat[1]), int(flat[2])), self._energy_from_bin(int(flat[3]))
                return int(flat[0]), None, None
            # Lists or tuples (length-4 structured, else scalar-like)
            if isinstance(action, (list, tuple)):
                if len(action) >= 4:
                    return int(action[0]), (int(action[1]), int(action[2])), self._energy_from_bin(int(action[3]))
                return int(action[0]), None, None
            # Scalars (int-like)
            return int(action), None, None
        except Exception as e:
            raise ValueError(f"Unable to normalize action: {action!r}") from e

    def _energy_from_bin(self, ebin: Optional[int]) -> Optional[int]:
        """Map an energy-bin index to a requested energy amount (e.g. ATTACK payload).

        Bin 0 means "use the action's default" (returns ``None``). Bins ``1..energy_bins-1``
        map linearly to 10%..100% of the configured max energy. The receiving action clamps
        the request to its own ``[min, available-energy]`` range.
        """
        if ebin is None or ebin <= 0:
            return None
        frac = min(1.0, ebin / float(max(1, self.energy_bins - 1)))
        return int(round(frac * self.config['max_energy']))


    def _is_action_valid_for_state(self, action: int, ship: dict, is_player: bool = True) -> Tuple[bool, str]:
        """
        Validate if an action is valid given the current game state.

        Enhanced action masking rules:
        1. DESTROYED state: only RESPAWN is valid
        2. Not DESTROYED: RESPAWN is invalid
        3. RECHARGING state: only WAIT and RECHARGE_END are valid
        4. Not RECHARGING: RECHARGE_END is invalid (WAIT is always valid for gaining action points)
        5. RECHARGING + full energy: only RECHARGE_END is valid
        6. ATTACK: requires enemy in same zone
        7. MINE: requires asteroid at current location
        8. SELL: requires trading post at current location
        9. All actions: respect energy requirements
        10. Energy-consuming actions masked when insufficient energy
        11. JUMP_TO_TRADING_POST and SELL: require nutrinium
        12. RAISE_SHIELDS: requires combat situation (enemy in same zone)
        13. JUMP_TO_ASTEROID: masked when already at asteroid with nutrinium >= 5%
        14. JUMP_TO_ASTEROID: masked when nearest asteroid is at same location (distance 0, would be a no-op)
        15. RECHARGE: masked when energy > 50% (avoid wasteful recharge cycles)
        16. JUMP_TO_TRADING_POST: masked when already at a trading post (use SELL)
        17. WAIT: masked when energy is critically low (< min useful cost) and NOT recharging
            (prevents dead-end: WAIT doesn't restore energy, only RECHARGE does)

        Args:
            action: The action to validate
            ship: The ship attempting the action
            is_player: Whether this is the player ship

        Returns:
            (is_valid, reason) - True if valid, False with reason string if invalid
        """
        return action_masker.is_action_valid(action, self._build_mask_state(ship, is_player=is_player))

    def _build_mask_state(self, ship: dict, is_player: bool = True) -> 'action_masker.MaskState':
        """Adapt the full simulator state into a neutral :class:`MaskState`.

        Resolves the candidate ATTACK/RAISE_SHIELDS/PLUNDER target list the same
        way the validity rules historically did: the player targets opponents;
        an opponent targets the player plus the other opponents.
        """
        if is_player:
            enemies = list(self.opponent_ships)
        else:
            enemies = [self.player_ship] + [s for s in self.opponent_ships if s is not ship]
        shield = ship.get('shield') if isinstance(ship.get('shield'), dict) else {}
        objective = (ship.get('objectives') or {}).get('negotiate') or {}
        return action_masker.MaskState(
            x=ship['x'],
            y=ship['y'],
            energy=ship['energy'],
            health=ship.get('health', 0),
            nutrinium=ship.get('nutrinium', 0),
            credits=ship.get('credits', 0),
            destroyed=ship.get('destroyed', False),
            recharging=ship.get('recharging', False),
            just_recharged=ship.get('just_recharged', False),
            shield_state=self._shield_state(ship),
            shield_value=shield.get('value', 0),
            shield_capacity=shield.get('capacity', 0),
            shields_up=ship.get('shields_up', False),
            modules=ship.get('modules') or [],
            negotiate_post_id=objective.get('tradingPostId'),
            enemies=enemies,
            asteroids=self.asteroids,
            trading_posts=self.trading_posts,
            wreckage=self.wreckage,
            map_width=self.map_width,
            map_height=self.map_height,
            max_energy=self.config['max_energy'],
            max_health=self.config['max_health'],
            energy_costs=self.config['energy_costs'],
            salvage_energy_cost=self.config['salvage']['energy_cost'],
            repair_cost=self.config['market']['repair'],
            action_restrictions=self.config.get('action_restrictions', {}),
        )

    def _get_action_mask(self, ship: dict = None, is_player: bool = True) -> np.ndarray:
        """
        Generate action mask for valid actions given the current game state.

        Args:
            ship: The ship to generate mask for (defaults to player_ship)
            is_player: Whether this ship is the player (affects target selection for ATTACK etc.)

        Returns:
            Boolean array where True means action is valid, False means invalid
        """
        if ship is None:
            ship = self.player_ship

        return action_masker.get_action_mask(self._build_mask_state(ship, is_player=is_player))
