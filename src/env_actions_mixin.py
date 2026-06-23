"""Action-execution and combat mixin for :class:`ProspectorsPiratesEnv`.

Contains the action dispatcher (:meth:`_execute_action`), every concrete
``_action_*`` handler, the shared combat helpers (hit chance, same-zone
targeting, combat-state bookkeeping) and the per-ship pre/post action ticks.
"""

from env_common import *


class EnvActionsMixin:
    """Concrete action handlers, combat resolution and tick bookkeeping."""

    def _execute_action(self, action: int, ship: dict, is_player: bool = True,
                        target: Optional[Tuple[int, int]] = None,
                        energy: Optional[int] = None) -> Tuple[float, dict]:
        """Execute an action for a ship.

        ``target`` (an explicit (x, y) coordinate) and ``energy`` (an explicit energy/payload
        amount) come from the structured action space. When ``None`` the action falls back to
        its automatic target selection / default energy, preserving the behaviour used by the
        heuristic AIs and legacy scalar-action callers.
        """
        reward = 0.0
        info = {'action': ActionType(action).name, 'success': False}

        if action == ActionType.WAIT:
            info['success'] = True

        elif action == ActionType.MINE:
            reward += self._apply_action_result(self._action_mine(ship), info)

        elif action in [ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                       ActionType.MOVE_EAST, ActionType.MOVE_WEST]:
            reward += self._apply_action_result(self._action_move(ship, action), info)

        elif action == ActionType.RECHARGE:
            info['success'] = self._action_recharge(ship)

        elif action == ActionType.RECHARGE_END:
            info['success'] = self._action_recharge_end(ship)

        elif action == ActionType.RAISE_SHIELDS:
            info['success'] = self._action_raise_shields(ship)

        elif action == ActionType.ATTACK:
            reward += self._apply_action_result(
                self._action_attack(ship, is_player, payload=energy, target_coord=target), info)

        elif action == ActionType.JUMP_TO_ASTEROID:
            reward += self._apply_action_result(self._action_jump(ship, target=target), info)

        elif action == ActionType.JUMP_TO_TRADING_POST:
            reward += self._apply_action_result(self._action_jump_to_trading_post(ship), info)

        elif action == ActionType.SELL:
            reward += self._apply_action_result(self._action_sell(ship), info)

        elif action == ActionType.RESPAWN:
            # Store state before respawn to calculate cost
            credits_before = ship.get('credits', 0)
            team_id_before = ship.get('team_id', 0)
            team_count_before = self.team_respawn_counts.get(team_id_before, 0)

            success = self._action_respawn(ship)
            info['success'] = success

            if success:
                # Calculate actual cost paid (capped by the score floor)
                credits_after = ship.get('credits', 0)
                credits_paid = credits_before - credits_after

                # Full insurance cost that should have been charged this respawn
                ti = self.config['team_insurance']
                respawn_cost = int(round(ti['base_cost_per_member'] * (1.0 + ti['cost_escalation'] * team_count_before)))
                insurance_covered = respawn_cost - credits_paid

                # Reward is negative based on credits lost (not fixed penalty)
                # Scale by a small factor to not overwhelm other rewards
                reward -= credits_paid * 0.1  # 10% of credits lost as negative reward

                # Add payload with respawn details
                info['payload'] = {
                    'respawn_cost': respawn_cost,
                    'credits_paid': credits_paid,
                    'insurance_covered': insurance_covered,
                    'respawn_count': ship.get('respawn_count', 0)
                }

        elif action == ActionType.PLUNDER:
            reward += self._apply_action_result(
                self._action_plunder(ship, is_player, target_coord=target), info)

        elif action == ActionType.SALVAGE:
            reward += self._apply_action_result(self._action_salvage(ship), info)

        elif action == ActionType.REPAIR:
            reward += self._apply_action_result(self._action_repair(ship), info)

        elif action == ActionType.NEGOTIATE:
            reward += self._apply_action_result(self._action_negotiate(ship), info)

        elif action == ActionType.LOWER_SHIELDS:
            info['success'] = self._action_lower_shields(ship)

        return reward, info

    def _apply_action_result(self, result, info: dict) -> float:
        """Unpack a '__action__' return value into 'info' and return its reward.

        Handlers return either `(reward, success)` or `(reward, success, payload)`;
        this consolidates the shared unpacking the dispatcher used to repeat per branch.
        """
        if isinstance(result, tuple) and len(result) == 3:
            r, success, payload = result
            info['payload'] = payload
        else:
            r, success = result
        info['success'] = success
        return r

    def _action_mine(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Mine an asteroid at the current location (spec-faithful).

        Success probability is the asteroid's nutrinium concentration (nutrinium/mass,
        0-1), lifted toward 1 by the miner's mine_accuracy skill. Payout is a biased-low
        random draw bounded by payout_modifier * remaining_nutrinium * yield_bonus, with a
        floor of 1 and capped at the remaining nutrinium. On success the asteroid loses
        'payout' nutrinium and exactly 1 mass; on failure it loses 1 mass only.
        """
        # Action restrictions: MINE is disallowed while recharging or with shields POWERED
        # (default). Honors metadata.actionRestrictions so configs can override.
        if not self._action_allowed(ship, 'MINE'):
            return -0.1, False

        energy_cost = max(0, self.config['energy_costs']['mine'] - self.skill(ship, 'mine_cost'))
        if ship['energy'] < energy_cost:
            return -0.1, False

        # Find asteroid at current location
        asteroid = self.get_entity_at_location(ship['x'], ship['y'], self.asteroids)
        if asteroid is None:
            return -0.1, False

        # Capture pre-mine state for detailed reporting
        energy_before = ship['energy']
        asteroid_mass_before = asteroid['mass']
        asteroid_nutr_before = asteroid['nutrinium']

        ship['energy'] -= energy_cost

        # Base success = nutrinium concentration (nutrinium / mass), clamped to [0, 1].
        # */0 mass (or 0 nutrinium) -> 0% success.
        if asteroid_mass_before <= 0 or asteroid_nutr_before <= 0:
            base_success = 0.0
        else:
            base_success = min(1.0, asteroid_nutr_before / asteroid_mass_before)

        # mine_accuracy skill closes the gap to certainty: Success = Base + (1-Base)*(skill*0.05)
        mine_accuracy = self.skill(ship, 'mine_accuracy')
        success_chance = base_success + (1.0 - base_success) * (mine_accuracy * 0.05)
        success_chance = max(0.0, min(1.0, success_chance))

        # Build detailed mining payload
        mine_details = {
            'asteroid_x': asteroid['x'],
            'asteroid_y': asteroid['y'],
            'ast_mass': f"{asteroid_mass_before}",
            'ast_nutr': f"{asteroid_nutr_before}",
            'ast_density': round(base_success, 4),
            'success_chance': round(success_chance * 100, 1),
            'mine_accuracy': mine_accuracy,
            'mine_yield': self.skill(ship, 'mine_yield_multiplier'),
            'mine_cost_skill': self.skill(ship, 'mine_cost'),
            'energy': f"{energy_before}->{ship['energy']}",
            'energy_cost': energy_cost,
        }

        if random.random() < success_chance:
            # Payout ceiling = payout_modifier * remaining nutrinium * (1 + mine_yield*0.1)
            remaining = asteroid_nutr_before
            mine_yield = self.skill(ship, 'mine_yield_multiplier')
            ceiling = (self.config['mining']['payout_modifier']
                       * remaining * (1.0 + mine_yield * 0.1))
            # Biased-low random draw via Beta(alpha<beta); min 1, capped at remaining.
            fraction = random.betavariate(
                self.config['mining']['payout_beta_alpha'],
                self.config['mining']['payout_beta_beta'],
            )
            payout = int(round(fraction * ceiling))
            payout = max(self.config['mining']['min_payout'], payout)
            payout = min(payout, remaining)

            asteroid['nutrinium'] -= payout
            asteroid['mass'] -= 1  # mass drops by exactly 1 per mine, regardless of payout
            if asteroid['mass'] < 0:
                asteroid['mass'] = 0
            ship['nutrinium'] += payout

            mine_details['payout'] = payout
    mine_details['ast_mass_after'] = asteroid['mass']
    mine_details['ast_nutr_after'] = asteroid['nutrinium']
    mine_details['ship_nutr'] = ship['nutrinium']

    return payout * 0.05, True, mine_details  # Reward for mining (precursor to SELL for credits)
    else:
        # Failed mining: asteroid still loses 1 mass.
        asteroid['mass'] -= 1
        if asteroid['mass'] < 0:
            asteroid['mass'] = 0

    mine_details['payout'] = 0
    mine_details['ast_mass_after'] = asteroid['mass']
    mine_details['ast_nutr_after'] = asteroid['nutrinium']

    return -0.05, False, mine_details

    def _action_move(self, ship: dict, action: int) -> Tuple[float, bool, Optional[dict]]:
        """Move ship in a direction (server axis: N=y+1, S=y-1, E=x+1, W=x-1)."""
        if ship['energy'] < self.config['energy_costs']['move']:
            return -0.1, False

        old_x, old_y = ship['x'], ship['y']
        new_x, new_y = old_x, old_y

        if action == ActionType.MOVE_NORTH:
            new_y = min(self.map_height - 1, old_y + 1)
        elif action == ActionType.MOVE_SOUTH:
            new_y = max(0, old_y - 1)
        elif action == ActionType.MOVE_EAST:
            new_x = min(self.map_width - 1, old_x + 1)
        elif action == ActionType.MOVE_WEST:
            new_x = max(0, old_x - 1)

        if new_x == old_x and new_y == old_y:
            return -0.1, False, {'from': (old_x, old_y), 'to': (new_x, new_y)}  # Tried to move off map

        ship['x'] = new_x
        ship['y'] = new_y
        ship['energy'] -= self.config['energy_costs']['move']

        return -0.01, True, {'from': (old_x, old_y), 'to': (new_x, new_y)}

    def _action_recharge(self, ship: dict) -> bool:
        """Start recharging"""
        if ship['recharging']:
            return False
        ship['recharging'] = True
        return True

    def _action_recharge_end(self, ship: dict) -> bool:
        """Stop recharging"""
        if not ship['recharging']:
            return False
        ship['recharging'] = False
        ship['just_recharged'] = True  # Prevent immediate re-recharging
        return True

    def _action_raise_shields(self, ship: dict) -> bool:
        """Raise shields toward full capacity (spec-faithful, value-based).

        Charges the shield's `value` toward its capacity in increments of
        shield_recharge_rate. Each increment costs (1 - shield_efficiency*0.05) energy.
        If the ship can afford the full charge it reaches POWERED; if energy runs out first it gets a partial charge and is left DRAINING (it will decay each tick).
        """
        if not self._action_allowed(ship, 'RAISE_SHIELDS'):
            return False

        shield = ship.get('shield')
        if not isinstance(shield, dict):
            return False

        rate = self.config['combat']['shield_recharge_rate']
        capacity = shield.get('capacity', self.config['combat']['base_shield_capacity'])
        current = shield.get('value', 0)
        delta = capacity - current

        # Already fully powered -- nothing to do.
        if delta <= 0 and shield.get('state') == 'POWERED':
            return False

        efficiency_factor = max(0.0, 1.0 - self.skill(ship, 'shield_efficiency') * 0.05)
        needed_chunks = max(1, math.ceil(delta / rate)) if delta > 0 else 0
        full_cost = int(round(needed_chunks * efficiency_factor))

        energy = ship.get('energy', 0)

        if delta <= 0 or energy >= full_cost:
            # Full raise: shield reaches capacity and becomes POWERED.
            shield['value'] = capacity
            shield['state'] = 'POWERED'
            ship['energy'] = max(0, energy - full_cost)
        else:
            # Partial raise: charge as many increments as affordable, then DRAINING.
            if efficiency_factor <= 0:
                affordable_chunks = needed_chunks
            else:
                affordable_chunks = int(energy // efficiency_factor)
        return False
            value_added = min(delta, affordable_chunks * rate)
            cost = int(round(affordable_chunks * efficiency_factor))
            shield['value'] = current + value_added
            shield['state'] = 'DRAINING'
            ship['energy'] = max(0, energy - cost)

        # Legacy alias + combat engagement (shields are a same-zone action).
        ship['shields_up'] = True
        ship['in_combat'] = True
        ship.setdefault('combat_opponent_positions', set()).add((ship['x'], ship['y']))

        return True

    def _action_lower_shields(self, ship: dict) -> bool:
        """Lower shields: a POWERED shield stops being maintained and begins DRAINING.

        The stored value is not discarded immediately; it decays by shield_recharge_rate
        each tick (handled in _post_action_tick) until the shield is DOWN.
        """
        shield = ship.get('shield')
        if not isinstance(shield, dict):
            return False
        if shield.get('state') == 'DOWN':
            return False
        shield['state'] = 'DRAINING'
        ship['shields_up'] = False
        return True

    def _max_jump_distance(self, ship: dict) -> float:
        """Maximum jump range = base max_jump_distance + jump_distance_skill * 10."""
        return self.config['max_jump_distance'] + self._skill(ship, 'jump_distance') * 10

    def _jump_energy_cost(self, ship: dict, distance: float) -> int:
        """Energy cost of a jump of the given Euclidean distance.

        cost = max(adjMinCost, round(jump_unit_cost * distance)) where
        adjMinCost = max(0, jump_min_cost - jump_cost_skill * 5). The skill lowers
        the floor (cheaper short jumps); per-unit cost grows with distance.
        """
        unit_cost = self.config['energy_costs']['jump']
        adj_min_cost = max(0, self.config['energy_costs']['jump_min_cost'] -
                           self._skill(ship, 'jump_cost') * 5)
        return int(max(adj_min_cost, round(unit_cost * distance)))

    def _action_jump(self, ship: dict, target: Optional[Tuple[int, int]] = None) -> Tuple[float, bool, Optional[dict]]:
        """Jump to a coordinate within the ship's max jump range.

        When an explicit `target` coordinate is supplied (structured action space) the ship
        jumps there directly. With no target it falls back to the best asteroid by
        nutrinium-to-distance score, so the model's observation of top asteroids aligns with
        where an auto JUMP goes (legacy behaviour).
        """
        if not self.action_allowed(ship, 'JUMP'):
            return -0.1, False, None

        if not self.has_module(ship, 'JUMP'):
            return -0.1, False, {'error': 'JUMP module not equipped'}

        if target is not None:
            # Explicit coordinate jump: clamp the requested target onto the map.
            tx = int(max(0, min(self.map_width - 1, target[0])))
            ty = int(max(0, min(self.map_height - 1, target[1])))
        else:
            # Auto-target: best asteroid by score (nutrinium value vs distance), not just nearest.
            top = self.get_top_asteroids(ship['x'], ship['y'], count=1)
            if not top:
                return -0.1, False, None
            best = top[0]
            tx, ty = best['x'], best['y']

        distance = self._calculate_distance(ship['x'], ship['y'], tx, ty)

        # Prevent jumping to same location (distance 0) -- this is a no-op
        if distance == 0:
            return -0.1, False, {'error': 'target is at current location'}

        # Range check: jumps beyond the ship's max range are rejected.
        if distance > self.max_jump_distance(ship):
            return -0.1, False, {'error': 'OUT_OF_JUMP_RANGE', 'distance': distance,
                                'max_jump': self._max_jump_distance(ship)}

        energy_cost = self._jump_energy_cost(ship, distance)

        if ship['energy'] < energy_cost:
            return -0.1, False, None

        # Jump to the target coordinate
        old_x, old_y = ship['x'], ship['y']
        ship['x'] = tx
        ship['y'] = ty
        ship['energy'] -= energy_cost

        payload = {'from': (old_x, old_y), 'to': (ship['x'], ship['y']), 'distance': distance, 'energy_cost': energy_cost}
        return -0.01, True, payload

    def _action_jump_to_trading_post(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Jump to nearest trading post"""
        if not self.action_allowed(ship, 'JUMP'):
            return -0.1, False, None
    if not self._has_module(ship, 'JUMP'):
        return -0.1, False, {'error': 'JUMP module not equipped'}

    # Find nearest trading post
    nearest_post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
    if nearest_post is None:
        return -0.1, False, None

    distance = self._calculate_distance(ship['x'], ship['y'], nearest_post['x'], nearest_post['y'])

    # Prevent jumping to same location (distance 0) -- should SELL instead
    if distance == 0:
        return -0.1, False, {'error': 'already at trading post, use SELL instead'}

    # Range check: jumps beyond the ship's max range are rejected.
    if distance > self._max_jump_distance(ship):
        return -0.1, False, {'error': 'OUT_OF_JUMP_RANGE', 'distance': distance,
                             'max_jump': self._max_jump_distance(ship)}

    energy_cost = self._jump_energy_cost(ship, distance)

    if ship['energy'] < energy_cost:
        return -0.1, False, None

    # Jump to trading post
    old_x, old_y = ship['x'], ship['y']
    ship['x'] = nearest_post['x']
    ship['y'] = nearest_post['y']
    ship['energy'] -= energy_cost

    payload = {'from': (old_x, old_y), 'to': (ship['x'], ship['y']), 'distance': distance, 'energy_cost': energy_cost}
    return -0.01, True, payload

    def action_sell(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Sell all nutrinium at a trading post using the dynamic market price.

        The sale price is the current market price lifted by the seller's team bonus.
        Each sale dips the market price (recovering over subsequent ticks), modelling
        supply pressure when many ships sell at once.
        """
        if not self._action_allowed(ship, 'SELL'):
            return -0.1, False, None

        trading_post = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
        if trading_post is None:
            return -0.1, False, None

        if ship['nutrinium'] <= 0:
            return -0.1, False, None

        nutrinium_sold = ship['nutrinium']
        base_price = self.market_price
        team_bonus = self.team_bonuses.get(ship.get('team_id', 0), 0.0)
        effective_price = base_price * (1.0 + team_bonus)
        credits_earned = int(round(nutrinium_sold * effective_price))

        ship['nutrinium'] = 0
        ship['credits'] += credits_earned

        # Sale pressure: dip the market price, floored at a fraction of the base price.
        base = self.config['market']['sell_nutrinium']
        floor = base * self.config['market']['price_min_factor']
        self.market_price = max(floor, self.market_price * (1.0 - self.config['market']['price_dip_per_sale']))

        payload = {
            'nutrinium_sold': nutrinium_sold,
            'credits_earned': credits_earned,
            'unit_price': round(effective_price, 2),
            'team_bonus': round(team_bonus, 4),
            'market_price_after': round(self.market_price, 2),
        }
        return credits_earned * 0.5, True, payload

    def _update_market(self) -> None:
        """Recover the market price toward its base each tick (price elasticity)."""
        base = self.config['market']['sell_nutrinium']
        rate = self.config['market']['price_recovery_rate']
        self.market_price += (base - self.market_price) * rate

    def _combat_hit_chance(self, attacker: dict, target: dict) -> float:
        """Shared roll-to-hit probability for ATTACK and PLUNDER.

        Base target number raised by attack_accuracy, lowered by the target's evade,
        eased when the target is recharging, then clamped to leave guaranteed miss/hit
        margins.
        """
        cfg = self.config['combat']
        chance = (cfg['base_target_number']
                  + self._skill(attacker, 'attack_accuracy') * 0.05
                  - self._skill(target, 'evade') * 0.05)
        if target.get('recharging'):
            chance += cfg['recharge_penalty']
        return max(cfg['guaranteed_hit_chance'],
                   min(1.0 - cfg['guaranteed_miss_chance'], chance))

    def _same_zone_enemies(self, ship: dict, is_player: bool) -> List[dict]:
        """Active (non-destroyed) enemy ships sharing this ship's tile."""
        if is_player:
            candidates = list(self.opponent_ships)
    def _action_plunder(self, ship: dict, is_player: bool,
                        target_coord: Optional[Tuple[int, int]] = None) -> Tuple[float, bool, Optional[dict]]:
        """Steal nutrinium from a shields-down enemy in the same zone.

        Only targets whose shields are DOWN can be plundered. The fixed plunder cost
        is paid even on a miss; on a hit a random amount (1..target nutrinium) is transferred
        to the attacker. An explicit ``target_coord`` (structured action space) prefers a
        plunderable enemy located there; otherwise the richest in-zone target is chosen.
        """
        if not self._action_allowed(ship, 'PLUNDER'):
            return -0.1, False, None

        cost = self.config['energy_costs']['plunder']
        if ship.get('energy', 0) < cost:
            return -0.1, False, None

        # Eligible targets: same-zone, shields DOWN, carrying nutrinium.
        targets = [t for t in self._same_zone_enemies(ship, is_player)
                   if self._shield_state(t) == 'DOWN' and t.get('nutrinium', 0) > 0]
        if not targets:
            return -0.1, False, {'error': 'no plunderable target in zone'}

        # Prefer a target at the requested coordinate, else go for the richest target.
        target = None
        if target_coord is not None:
            tx, ty = int(target_coord[0]), int(target_coord[1])
            at_coord = [t for t in targets if t['x'] == tx and t['y'] == ty]
            if at_coord:
                target = max(at_coord, key=lambda t: t.get('nutrinium', 0))
        if target is None:
            target = max(targets, key=lambda t: t.get('nutrinium', 0))

        ship['energy'] -= cost
        ship['in_combat'] = True
        ship.setdefault('combat_opponent_positions', set()).add((target['x'], target['y']))

        hit_chance = self._combat_hit_chance(ship, target)
        details = {
            'target': target.get('name', 'Unknown'),
            'hit_chance': round(hit_chance * 100, 1),
            'target_nutrinium': target.get('nutrinium', 0),
            'energy_cost': cost,
        }
        if random.random() >= hit_chance:
            details['hit'] = False
            return -0.05, False, details

        stolen = random.randint(1, target['nutrinium'])
        target['nutrinium'] -= stolen
        ship['nutrinium'] = ship.get('nutrinium', 0) + stolen

        details['hit'] = True
        details['stolen'] = stolen
        return stolen * 0.1, True, details

    def _action_salvage(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Recover nutrinium from wreckage at the current location (SALVAGE module).

        Pulls a random amount from wreckage on the ship's tile, scaled up by the
        salvage_yield skill and capped by what remains. Requires the SALVAGE module.
        """
        if not self._action_allowed(ship, 'SALVAGE'):
            return -0.1, False, None

        if not self.has_module(ship, 'SALVAGE'):
            return -0.1, False, {'error': 'SALVAGE module not equipped'}

        cost = self.config['salvage']['energy_cost']
        if ship.get('energy', 0) < cost:
            return -0.1, False, None

        wreck = next((w for w in self.wreckage
                      if w['x'] == ship['x'] and w['y'] == ship['y'] and w.get('nutrinium', 0) > 0), None)
        if wreck is None:
            return -0.1, False, {'error': 'no wreckage here'}

        ship['energy'] -= cost
        remaining = wreck['nutrinium']
        base_recover = random.randint(1, remaining)
        recovered = min(remaining, int(round(base_recover * (1.0 + self._skill(ship, 'salvage_yield') * 0.1))))
        recovered = max(1, recovered)

        wreck['nutrinium'] -= recovered
        ship['nutrinium'] = ship.get('nutrinium', 0) + recovered
        if wreck['nutrinium'] <= 0:
            self.wreckage.remove(wreck)

        details = {'recovered': recovered, 'wreckage_remaining': wreck['nutrinium'] if wreck in self.wreckage else 0,
                   'energy_cost': cost}
        return recovered * 0.1, True, details

    def _action_repair(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Restore hull to full at a trading post (REPAIR module).
    if not self._action_allowed(ship, 'REPAIR'):
        return -0.1, False, None
    
    if not self._has_module(ship, 'REPAIR'):
        return -0.1, False, {'error': 'REPAIR module not equipped'}
    
    trading_post = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
    if trading_post is None:
        return -0.1, False, {'error': 'not at a trading post'}
    
    cost = self.config['market']['repair']
    if ship.get('credits', 0) < cost:
        return -0.1, False, {'error': 'insufficient credits to repair'}
    
    health_before = ship.get('health', 0)
    ship['credits'] -= cost
    ship['health'] = self.config['max_health']
    
    details = {'health': f"{health_before}->{ship['health']}", 'credit_cost': cost}
    return 0.05, True, details
    
    def action_negotiate(self, ship: dict) -> Tuple[float, bool, Optional[dict]]:
        """Negotiate a team bonus at the ship's objective trading post.

        Resolves to SUCCESS / FAIL / NEUTRAL using base odds shifted by negotiate_skill.
        SUCCESS raises the team's bonus (amplified by negotiate_ambition, diminishing
        toward the ceiling); FAIL lowers it (softened by negotiate_caution). The
        objective is then consumed and a new one assigned.

        """
        if not self.action_allowed(ship, 'NEGOTIATE'):
            return -0.1, False, None
    
        cost = self.config['energy_costs']['negotiate']
        if ship.get('energy', 0) < cost:
            return -0.1, False, None
    
        objective = (ship.get('objectives') or {}).get('negotiate')
        post = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
        if post is None or objective is None or post.get('id') != objective.get('tradingPostId'):
            return -0.1, False, {'error': 'not at negotiate objective'}
    
        ship['energy'] -= cost
        ncfg = self.config['negotiate']
        skill = self.skill(ship, 'negotiate_skill')
        succ = max(0.0, min(1.0, ncfg['base_success_chance'] + skill * 0.05))
        fail = max(ncfg['min_fail_chance'], ncfg['base_fail_chance'] - skill * 0.05)
        if succ + fail > 1.0:
            fail = max(0.0, 1.0 - succ)
    
        team_id = ship.get('team_id', 0)
        current = self.team_bonuses.get(team_id, 0.0)
        max_bonus = ncfg['max_team_bonus']
    
        roll = random.random()
        if roll < succ:
            outcome = 'SUCCESS'
            gain = ncfg['bonus_gain'] * (1.0 + self.skill(ship, 'negotiate_ambition') * 0.1)
            # Diminishing returns toward the ceiling.
            applied = gain * (1.0 - current / max_bonus) if max_bonus > 0 else gain
            self.team_bonuses[team_id] = min(max_bonus, current + applied)
            reward = 0.1
        elif roll < succ + fail:
            outcome = 'FAIL'
            penalty = ncfg['bonus_penalty'] * max(0.0, 1.0 - self.skill(ship, 'negotiate_caution') * 0.08)
            self.team_bonuses[team_id] = max(0.0, current - penalty)
            reward = -0.05
        else:
            outcome = 'NEUTRAL'
            reward = 0.0
    
        # Consume the objective and assign a fresh one.
        new_objective = None
        if self.trading_posts:
            new_post = random.choice(self.trading_posts)
            new_objective = {'tradingPostName': new_post.get('name'), 'tradingPostId': new_post.get('id')}
            ship.setdefault('objectives', {})[negotiate] = new_objective
    
        details = {
            'outcome': outcome,
            'team_id': team_id,
            'team_bonus': round(self.team_bonuses[team_id], 4),
            'energy_cost': cost,
        }
        return reward, outcome != 'FAIL', details
    
    def action_respawn(self, ship: dict) -> bool:
        """Respawn a destroyed ship using escalating team insurance.

        Cost per respawn = base_cost_per_member * (1 + cost_escalation * team_respawn_count),
        charged to the ship's credits but never driving the score below zero (the
        team insurance absorbs the shortfall). The team's respawn counter then increments.
        """
        if not ship.get('destroyed', False):
            return False
    
        team_id = ship.get('team_id', 0)
        team_count = self.team_respawn_counts.get(team_id, 0)
    base = self.config['team_insurance']['base_cost_per_member']
    escalation = self.config['team_insurance']['cost_escalation']
    respawn_cost = int(round(base * (1.0 + escalation * team_count)))
    
    # Score floor: credits never go negative (insurance covers any shortfall).
    ship['credits'] = max(0, ship.get('credits', 0) - respawn_cost)
    
    # Reset ship state
    ship['destroyed'] = False
    ship['health'] = self.config['max_health']
    ship['energy'] = 0 # Start with 0 energy after respawn
    ship['nutrinium'] = 0 # Lose all nutrinium
    ship['shields_up'] = False
    ship['recharging'] = False
    ship['state'] = 'READY'
    if isinstance(ship.get('shield'), dict):
        ship['shield']['value'] = 0
        ship['shield']['state'] = 'DOWN'
    
    # Respawn at random location
    ship['x'] = random.randint(0, self.map_width - 1)
    ship['y'] = random.randint(0, self.map_height - 1)
    
    # Increment respawn counters (team-shared + per-ship)
    self.team_respawn_counts[team_id] = team_count + 1
    ship['respawn_count'] = ship.get('respawn_count', 0) + 1
    
    return True
    
    def _action_attack(self, ship: dict, is_player: bool, payload: Optional[int] = None,
                       target_coord: Optional[Tuple[int, int]] = None) -> Tuple[float, bool, Optional[dict]]:
        """Attack an enemy ship in the same zone.

        Honors actionRestrictions, commits the energy payload even on a miss, applies
        the value-based shield hold/break model, and creates wreckage on a kill. The optional
        `target_coord` lets a structured action prefer a specific in-zone enemy; `payload`
        (from the energy bin) sets the attack energy, clamped to [attack_min, available].
        """
        if not self._action_allowed(ship, 'ATTACK'):
            return -0.1, False, None
    
        if ship['energy'] < self.config['energy_costs']['attack']:
            return -0.1, False, None
    
        # Determine targets depending on who is attacking
        if is_player:
            # Select weakest active opponent in the same zone
            same_zone_targets = [
                t for t in self.opponent_ships
                if not t.get('destroyed', False)
                and t['x'] == ship['x'] and t['y'] == ship['y']
            ]
            if not same_zone_targets:
                return -0.1, False, None
    
            # compute raw score for sorting (reuse _calculate_enemy_combat_score with raw=True)
            scored = [(self._calculate_enemy_combat_score(t, raw=True), t) for t in same_zone_targets]
            scored.sort(key=lambda x: x[0])  # ascending -> weakest first
            target = scored[0][1]
    
        else:
            # Opponent attacking: target any ship in the same zone (player or other opponents)
            # Build list of all potential targets in the same zone
            same_zone_targets = []
    
            # Consider the player as a target
            if (not self.player_ship.get('destroyed', False)
                and self.player_ship['x'] == ship['x']
                and self.player_ship['y'] == ship['y']):
                same_zone_targets.append(self.player_ship)
    
            # Consider other opponents as targets
            for other in self.opponent_ships:
                if other is ship:
                    continue  # Don't attack self
                if other.get('destroyed', False):
                    continue
                if other['x'] == ship['x'] and other['y'] == ship['y']:
                    same_zone_targets.append(other)
    
            if not same_zone_targets:
                return -0.1, False, None
    
            # Select the weakest target (lowest combat score)
            scored = [(self._calculate_enemy_combat_score(t, raw=True), t) for t in same_zone_targets]
            scored.sort(key=lambda x: x[0])  # ascending -> weakest first
            target = scored[0][1]
    
        # Structured action: prefer an in-zone enemy at the requested coordinate when one exists.
        if target_coord is not None and same_zone_targets:
            tx, ty = int(target_coord[0]), int(target_coord[1])
            preferred = [t for t in same_zone_targets if t['x'] == tx and t['y'] == ty]
            if preferred:
                pscored = [(self._calculate_enemy_combat_score(t, raw=True), t) for t in preferred]
                pscored.sort(key=lambda x: x[0])
                target = pscored[0][1]
    
        if target is None:
            return -0.1, False, None
    
        # Capture pre-combat state for detailed reporting
    target_health_before = target['health']
    target_shield = target.get('shield') if isinstance(target.get('shield'), dict) else None
    target_shield_state = self._shield_state(target)
    target_shield_value_before = target_shield['value'] if target_shield else 0
    target_energy_before = target.get('energy', 0)
    attacker_energy_before = ship['energy']
    
    cfg = self.config['combat']
    
    # Commit energy payload (consumed even on a miss). Until the Dict action space
    # exposes an energy bin (P9), use the configured default, capped by available energy.
    if payload is None:
        payload = cfg['default_attack_payload']
    payload = int(max(self.config['energy_costs']['attack'], min(payload, ship['energy'])))
    ship['energy'] -= payload
    
    # An attack engages both ships regardless of whether it lands.
    ship['in_combat'] = True
    ship.setdefault('combat_opponent_positions', set()).add((target['x'], target['y']))
    target['in_combat'] = True
    target.setdefault('combat_opponent_positions', set()).add((ship['x'], ship['y']))
    
    # --- Hit roll: base target number lifted by attack accuracy, lowered by evade,
    # eased when the target is recharging. Clamped to leave a guaranteed miss/hit margin.
    hit_chance = (cfg['base_target_number']
                 + self._skill(ship, 'attack_accuracy') * 0.05
                 - self._skill(target, 'evade') * 0.05)
    if target.get('recharging'):
        hit_chance += cfg['recharge_penalty']
    hit_chance = max(cfg['guaranteed_hit_chance'],
                     min(1.0 - cfg['guaranteed_miss_chance'], hit_chance))
    hit = random.random() < hit_chance
    
    combat_details = {
        'target': target.get('name', 'Unknown'),
        'atk_energy': f'{attacker_energy_before}->{ship["energy"]}',
        'payload': payload,
        'atk_power': self._skill(ship, 'attack_power'),
        'atk_accuracy': self._skill(ship, 'attack_accuracy'),
        'hit_chance': round(hit_chance * 100, 1),
        'def_evade': self._skill(target, 'evade'),
        'def_shield_state': target_shield_state,
        'def_shield_value': target_shield_value_before,
        'def_shield_str': self._skill(target, 'shield_strength'),
        'def_energy': target_energy_before,
    }
    
    if not hit:
        combat_details['hit'] = False
        combat_details['def_health'] = f"{target_health_before}->{target['health']}"
        # Missing still cost the payload; small penalty to discourage spray-and-pray.
        return -0.05, False, combat_details
    
    # --- Damage: payload scaled by attack_power (+10%/pt) and a random spread.
    avg_multiplier = 1.0 + self._skill(ship, 'attack_power') * 0.1
    variance = random.uniform(1.0 - cfg['damage_variance'], 1.0 + cfg['damage_variance'])
    damage = max(1, int(round(payload * avg_multiplier * variance)))
    
    # --- Shield resistance: base + shield_strength/20, capped at 0.75.
    resistance = min(0.75, cfg['base_shield_resistance']
                     + self._skill(target, 'shield_strength') * 0.05)
    shield_dmg = int(round(damage * cfg['attack_shield_damage']))
    
    if target_shield and target_shield_state == 'POWERED' and shield_dmg <= target_shield_value_before:
        # HOLD: shields absorb the strike; reduced damage bleeds through to the hull.
        health_dmg = round(damage * (1.0 - resistance))
        target_shield['value'] = target_shield_value_before - shield_dmg
    else:
        # BREAK (also covers shields DOWN / value 0): the portion the shield could
        # stop is reduced by resistance; the remainder lands at full strength.
        absorbable = math.ceil(target_shield_value_before / cfg['attack_shield_damage']) \
            if target_shield_value_before > 0 else 0
        absorbable = min(absorbable, damage)
        reduced = round(absorbable * (1.0 - resistance))
        unreduced = damage - absorbable
        health_dmg = reduced + unreduced
        if target_shield:
            target_shield['value'] = 0
            target_shield['state'] = 'DOWN'
        target['shields_up'] = False
    
    health_dmg = max(0, health_dmg)
    target['health'] = max(0, target['health'] - health_dmg)
    
    combat_details['hit'] = True
    combat_details['damage'] = damage
    combat_details['health_dmg'] = health_dmg
    combat_details['def_health'] = f"{target_health_before}->{target['health']}"
    combat_details['def_shield_value_after'] = target_shield['value'] if target_shield else 0
    
    # Check if target is destroyed
    if target['health'] <= 0:
        target['destroyed'] = True
        target['state'] = 'DESTROYED'
    
    # Destroyed ships spill part of their nutrinium into salvageable wreckage at
    # their location (recoverable via SALVAGE); it is NOT given to the attacker.
    nutrinium_before = target['nutrinium']
    wreckage_nutr = int(round(self.config['salvage']['wreckage_percent'] * nutrinium_before))
    if wreckage_nutr > 0:
    self.wreckage.append({'x': target['x'], 'y': target['y'], 'nutrinium': wreckage_nutr})
    
    # Wipe destroyed-ship resources and shields.
    target['energy'] = 0
    target['nutrinium'] = 0
    if target_shield:
        target_shield['value'] = 0
        target_shield['state'] = 'DOWN'
    target['shields_up'] = False
    
    reward = 0.5 + wreckage_nutr * 0.1  # kill reward; salvage realises the value
    
    combat_details['destroyed'] = True
    combat_details['wreckage_nutrinium'] = wreckage_nutr
    
    return reward, True, combat_details
    else:
        # Successful hit but target survived. Small reward; combat is a means to an end.
        combat_details['destroyed'] = False
        combat_details['target_health'] = target['health']
    
    return 0.02, True, combat_details
    
    def update_combat_states(self):
        """
        Update each ship's 'state' based on game state.

        COMBAT state should only be set when a ship is actively engaged in combat
        AND there is an enemy in the same zone. Ships attacked from a different zone
        should not remain in COMBAT state once the turn ends.

        This is called each step after all actions are executed.
        """
        # Build list of all ships (player + opponents)
        ships = [self.player_ship] + list(self.opponent_ships)
    
        for s in ships:
            if s is None:
                continue
    
            # Do not touch destroyed ships
            if s.get('destroyed', False):
                s['state'] = 'DESTROYED'
                continue
    
            # Recharging should remain RECHARGING
            if s.get('recharging', False):
                s['state'] = 'RECHARGING'
                continue
    
            # Combat state requires:
            # 1. The in_combat flag was set (an attack/shield action occurred)
            # 2. The combat was with an opponent at this ship's zone
            # 3. An enemy is STILL present in this zone (they may have moved away)
            combat_positions = s.get('combat_opponent_positions', set())
            was_same_zone_combat = s.get('in_combat', False) and (s['x'], s['y']) in combat_positions
    
            # Verify an enemy is still actually in the same zone
            if was_same_zone_combat:
                if s is self.player_ship:
                    enemies = self.opponent_ships
                else:
                    enemies = [self.player_ship] + [o for o in self.opponent_ships if o is not s]
                enemy_still_here = any(
                    e['x'] == s['x'] and e['y'] == s['y'] and not e.get('destroyed', False)
                    for e in enemies
                )
            else:
                enemy_still_here = False
    
            if was_same_zone_combat and enemy_still_here:
                s['state'] = 'COMBAT'
            else:
                # No same-zone combat this turn -> READY
                # Reset from any non-READY state (COMBAT, RECHARGING, etc.)
                if s.get('state', 'READY').upper() != 'READY':
                    s['state'] = 'READY'
    
        # Always clear combat flags for next turn
        s['in_combat'] = False
        s['combat_opponent_positions'] = set()
    
    def effective_max_energy(self, ship: dict) -> int:
        """Energy cap for a ship: base maxEnergy + energy max skill (+10 per point)."""
        return int(self.config['max_energy'] + self._skill(ship, 'energy_max') * 10)
    
    def _pre_action_tick(self, ship: dict) -> None:
        """Apply the pre-action portion of the per-ship tick order (steps 1-2).

        1. Shield maintenance: if shields are POWERED, deduct the per-tick
           maintenance cost. If energy drops below 0, shields transition to
           DRAINING and energy is clamped to 0.
        2. Recharge gain: if recharging, add energyPerRecharge + recharge_energy*2,
           capped at the ship's effective maximum energy. Applied BEFORE the action
           so the gained energy is spendable on the same tick.

        """
        if ship is None or ship.get('destroyed', False):
    return
    
    shield = ship.get('shield')
    # Step 1: shield maintenance (only while POWERED)
    if shield and shield.get('state') == 'POWERED':
        ship['energy'] -= self.config['energy_costs']['shield_maintenance']
        if ship['energy'] < 0:
            ship['energy'] = 0
            shield['state'] = 'DRAINING'
    
    # Step 2: recharge gain (before action)
    if ship.get('recharging', False):
        gain = self.config['energy_per_recharge'] + self._skill(ship, 'recharge_energy') * 2
        cap = self._effective_max_energy(ship)
        ship['energy'] = min(cap, ship['energy'] + gain)
    
    def _post_action_tick(self, ship: dict) -> None:
        """Apply the post-action portion of the per-ship tick order (step 4).

        4. Shield drain: if shields are DRAINING, their value decreases by
           shieldRechargeRate per tick until reaching 0 (DOWN).
        """
        if ship is None or ship.get('destroyed', False):
            return
        shield = ship.get('shield')
        if shield and shield.get('state') == 'DRAINING':
            shield['value'] -= self.config['combat']['shield_recharge_rate']
            if shield['value'] <= 0:
                shield['value'] = 0
                shield['state'] = 'DOWN'
                ship['shields_up'] = False