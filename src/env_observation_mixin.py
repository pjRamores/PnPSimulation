"""Observation-generation mixin for :class:`ProspectorsPiratesEnv`.

Holds the per-ship ModelSpec plumbing and the legacy full-observation builder
along with the top-asteroid / extreme-enemy / combat-score helpers and the
episode info dict.
"""

from env_common import *
from utils import action_masker


class EnvObservationMixin:
    """Observation construction and per-ship spec plumbing."""

    def _set_ship_model_spec(self, self, ship: dict, spec: Optional[ModelSpec] = None) -> None:
        """
        Associate a ModelSpec with a ship.

        Args:
            ship: Ship dictionary
            spec: ModelSpec instance (default = DEFAULT_FULL_SPEC)
        """
        if spec is None:
            spec = DEFAULT_FULL_SPEC
        self._ship_model_specs[id(ship)] = spec

    def _get_ship_model_spec(self, self, ship: dict) -> ModelSpec:
        """
        Get the ModelSpec for a ship, defaulting to full observation if not set.

        Args:
            ship: Ship dictionary

        Returns:
            ModelSpec instance
        """
        return self._ship_model_specs.get(id(ship), DEFAULT_FULL_SPEC)

    def _get_observation_generator(self, self, spec: ObservationSpec) -> ObservationGenerator:
        """
        Get or create an observation generator for a spec.

        Args:
            spec: ObservationSpec instance

        Returns:
            ObservationGenerator instance
        """
        spec_key = spec.observation_type
        if spec_key not in self._observation_generators:
            self._observation_generators[spec_key] = get_observation_generator(spec, self)
        return self._observation_generators[spec_key]

    def generate_observation_for_ship(self, self, ship: dict) -> Dict[str, np.ndarray]:
        """
        Generate observation for a ship using its assigned ModelSpec.

        Args:
            ship: Ship dictionary

        Returns:
            Dict observation with 'observation' and 'action_mask'
        """
        spec = self._get_ship_model_spec(ship)
        gen = self._get_observation_generator(spec.observation_spec)
        return gen.generate(ship)

    def _get_observation(self, self, skip_mask: bool = False, use_spec: bool = True) -> Dict[str, np.ndarray]:
        """Get the current observation with enhanced ship state and entity info.

        Args:
            skip_mask: If True, return a dummy action mask (all ones) to save computation.
                       Used for enemy observations where the mask is not needed.
            use_spec: If True, use the ship's assigned ModelSpec for observation generation.
                      If False, use the legacy full observation method.
        """
        # If using model spec system, delegate to generator (but only if not already generating)
        if use_spec and not getattr(self, '_generating_observation', False):
            return self._generate_observation_for_ship(self.player_ship)

        # Legacy observation generation (full format)
        obs = []
        ship = self.player_ship
        abilities = ship.get('abilities', {})
        max_abilities = self.config.get('abilities', {})

        # === ENHANCED SHIP STATE (24 values) ===
        # Basic stats (6 values)
        obs.extend([
            ship['x'] / max(1, self.map_width),
            ship['y'] / max(1, self.map_height),
            ship['energy'] / max(1, self.config['max_energy']),
            ship['health'] / max(1, self.config['max_health']),
            min(ship['nutrinium'], self.config['max_nutrinium_cargo']) / max(1, self.config['max_nutrinium_cargo']),
            min(ship['credits'], self.config['max_credits']) / max(1, self.config['max_credits']),
        ])

        # State flags (3 values)
        obs.extend([
            1.0 if ship.get('recharging', False) else 0.0,
        1.0 if ship.get('shields_up', False) else 0.0,
        1.0 if ship.get('state', 'READY') == 'READY' else 0.0,

        # Skill points (2 values)
        obs.extend([
            ship.get('skill_points_total', 5) / max(1, self.config.get('max_skill_points', 20)),
            ship.get('skill_points_spent', 0) / max(1, self.config.get('max_skill_points', 20)),
        ])

        # Abilities (12 values)
        obs.extend([
            abilities.get('energy_max', 5) / max(1, max_abilities.get('energy_max', 10)),
            abilities.get('recharge_energy', 0) / max(1, max_abilities.get('recharge_energy', 10)),
            abilities.get('mine_accuracy', 0) / max(1, max_abilities.get('mine_accuracy', 10)),
            abilities.get('mine_yield_multiplier', 1) / max(1, max_abilities.get('mine_yield_multiplier', 5)),
            abilities.get('mine_cost', 2) / max(1, max_abilities.get('mine_cost', 10)),
            abilities.get('combat_salvage_multiplier', 0) / max(1, max_abilities.get('combat_salvage_multiplier', 5)),
            abilities.get('sensor_range', 1) / max(1, self.config['sensor_range']),
            abilities.get('attack_accuracy', 0) / max(1, max_abilities.get('attack_accuracy', 10)),
            abilities.get('attack_power', 0) / max(1, max_abilities.get('attack_power', 10)),
            abilities.get('evade', 0) / max(1, max_abilities.get('evade', 10)),
            abilities.get('shield_strength', 0) / max(1, max_abilities.get('shield_strength', 10)),
            abilities.get('jump_distance', 0) / max(1, max_abilities.get('jump_distance', 10)),
        ])

        # Action counter (1 value) - normalized by max_steps (typical ~300)
        obs.append(self.action_counter / max(1, self.max_steps))

        # === STRATEGIC CONTEXT (8 values) ===
        # These high-signal features directly encode actionable state
        map_diag = max(1.0, math.sqrt(self.map_width**2 + self.map_height**2))

        # 1. At asteroid with nutrinium?
        ast_here = self.get_entity_at_location(ship['x'], ship['y'], self.asteroids)
        obs.append(1.0 if (ast_here and ast_here.get('nutrinium', 0) > 0) else 0.0)

        # 2. At trading post?
        tp_here = self.get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
        obs.append(1.0 if tp_here else 0.0)

        # 3. Cargo fullness (nutrinium as fraction of a "sell-worthy" amount ~25)
        obs.append(min(1.0, ship.get('nutrinium', 0) / 25.0))

        # 4. Enemy in same zone?
        enemy_here = any(
            e['x'] == ship['x'] and e['y'] == ship['y'] and not e.get('destroyed', False)
            for e in self.opponent_ships
        )
        obs.append(1.0 if enemy_here else 0.0)

        # 5-6. Direction to best asteroid (dx, dy normalized to [-1, 1])
        top_ast = self._get_top_asteroids(ship['x'], ship['y'], count=1)
        if top_ast:
            dx_ast = (top_ast[0]['x'] - ship['x']) / max(1, self.map_width)
            dy_ast = (top_ast[0]['y'] - ship['y']) / max(1, self.map_height)
        else:
            dx_ast, dy_ast = 0.0, 0.0
        obs.extend([dx_ast, dy_ast])

        # 7-8. Direction to nearest trading post (dx, dy normalized to [-1, 1])
        nearest_tp = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
        if nearest_tp:
            dx_tp = (nearest_tp['x'] - ship['x']) / max(1, self.map_width)
            dy_tp = (nearest_tp['y'] - ship['y']) / max(1, self.map_height)
        else:
            dx_tp, dy_tp = 0.0, 0.0
        obs.extend([dx_tp, dy_tp])

        # === LOCAL SENSOR GRID (with clamped/shifted window to maximize valid cells) ===
        sensor_range = self.config['sensor_range']
        side = 2 * sensor_range + 1  # Grid dimension (e.g., 11 for sensor_range=5)

        # Calculate top-left corner of a centered window
        x_min = ship['x'] - sensor_range
        y_min = ship['y'] - sensor_range

        # Clamp window to stay within map bounds (shifts window when near edges)
        # This maximizes the number of valid cells in the observation
        x_min = max(0, min(x_min, self.map_width - side)) if self.map_width >= side else 0
        y_min = max(0, min(y_min, self.map_height - side)) if self.map_height >= side else 0

        # Fill the sensor grid in row-major order (same as before for consistency)
        for row in range(side):
            for col in range(side):
                x = x_min + col
                y = y_min + row

                # Check if coordinate is valid (should almost always be true with clamping)
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    # Default: empty cell
                    entity_type = 0.0

                    # Player's own cell remains 0.0 (empty)
                    if x == ship['x'] and y == ship['y']:
                        entity_type = 0.0
                    # Check for entities (priority: enemy > trading_post > asteroid)
                    elif self.get_entity_at_location(x, y, self.opponent_ships):
                        entity_type = 1.0
                entity_type = 0.66
            elif self._get_entity_at_location(x, y, self.asteroids):
                entity_type = 0.33

            obs.append(entity_type)
        else:
            # Out of bounds (should be rare with clamping, only when map < sensor grid)
            obs.append(-1.0)

        # === TOP 5 ASTEROIDS (30 values: 5 asteroids * 6 features) ===
        top_asteroids = self._get_top_asteroids(ship['x'], ship['y'], count=self.config['top_asteroids_count'])
        max_dist = math.sqrt(self.map_width**2 + self.map_height**2)
        max_mass = float(self.config.get('asteroid_mass_max', 80))

        for asteroid in top_asteroids:
            obs.extend([
                asteroid['x'] / max(1, self.map_width),
                asteroid['y'] / max(1, self.map_height),
                asteroid['mass'] / max(1.0, max_mass),
                asteroid['nutrinium'] / max(1.0, max_mass),
                asteroid['distance'] / max(1.0, max_dist),
                asteroid['score'],  # Already normalized 0-1
            ])

        # Pad with zeros if fewer than 5 asteroids
        for _ in range(self.config['top_asteroids_count'] - len(top_asteroids)):
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # === NEAREST TRADING POST (3 values) ===
        nearest_post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
        if nearest_post:
            dist = self._calculate_distance(ship['x'], ship['y'], nearest_post['x'], nearest_post['y'])
            obs.extend([
                nearest_post['x'] / max(1, self.map_width),
                nearest_post['y'] / max(1, self.map_height),
                dist / max(1, max_dist),
            ])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # === TWO ENEMY TYPES (14 values: 2 enemies * 7 features) ===
        # Get strongest and weakest enemies at same coordinates as player
        strongest, weakest = self._get_extreme_enemies(ship['x'], ship['y'])

        for enemy in [strongest, weakest]:
            if enemy:
                combat_score = self._calculate_enemy_combat_score(enemy)
                obs.extend([
                    enemy['x'] / max(1, self.map_width),
                    enemy['y'] / max(1, self.map_height),
                    enemy['energy'] / max(1, self.config['max_energy']),
                    enemy['health'] / max(1, self.config['max_health']),
                    min(enemy['nutrinium'], 100) / 100.0,
                    min(enemy['credits'], 1000) / 1000.0,
                    combat_score,  # Already normalized 0-1
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # === SPEC-FIDELITY FEATURES (24 values; appended to keep legacy offsets stable) ===
        max_abil = self.config.get('abilities', {})

        # New skills beyond the legacy 12-ability block (7 values)
        obs.extend([
            abilities.get('shield_capacity', 0) / max(1, max_abil.get('shield_capacity', 10)),
            abilities.get('shield_efficiency', 0) / max(1, max_abil.get('shield_efficiency', 10)),
            abilities.get('jump_cost', 0) / max(1, max_abil.get('jump_cost', 10)),
            abilities.get('salvage_yield', 0) / max(1, max_abil.get('salvage_yield', 10)),
            abilities.get('negotiate_skill', 0) / max(1, max_abil.get('negotiate_skill', 10)),
            abilities.get('negotiate_cautious', 0) / max(1, max_abil.get('negotiate_cautious', 10)),
            abilities.get('negotiate_ambition', 0) / max(1, max_abil.get('negotiate_ambition', 10)),
        ])

        # Shield state machine (5 values): one-hot state + value fill + capacity
        shield = ship.get('shield') if isinstance(ship.get('shield'), dict) else {}
        sstate = self._shield_state(ship)
        scap = float(shield.get('capacity', 0) or 0)
        sval = float(shield.get('value', 0) or 0)
        max_capacity = self.config['combat']['base_shield_capacity'] + max_abil.get('shield_capacity', 10) * 10
        obs.extend([
            1.0 if sstate == 'POWERED' else 0.0,
            1.0 if sstate == 'DRAINING' else 0.0,
            1.0 if sstate == 'DOWN' else 0.0,
            (sval / scap) if scap > 0 else 0.0,
            scap / max(1.0, max_capacity),
        ])

        # Equipable modules one-hot (3 values)
        obs.extend([
            1.0 if self._has_module(ship, 'JUMP') else 0.0,
            1.0 if self._has_module(ship, 'REPAIR') else 0.0,
            1.0 if self._has_module(ship, 'SALVAGE') else 0.0,
        ])

        # Team + economy context (3 values)
        team_id = int(ship.get('team_id', 0) or 0)
        team_bonus = float(self.team_bonuses.get(team_id, 0)) if hasattr(self, 'team_bonuses') else 0.0
        market_ref = max(1.0, float(self.config['market']['sell_nutrinium']))
        obs.extend([
            min(1.0, team_id / 3.0),
        max(-1.0, min(1.0, team_bonus)),
        min(1.0, getattr(self, 'market_price', market_ref) / market_ref),
        ])

        # Negotiate objective trading post (3 values): present + direction
        objective = (ship.get('objectives') or {}).get('negotiate')
        obj_post = None
        if objective:
            obj_id = objective.get('tradingPostId')
            obj_post = next((p for p in self.trading_posts if p.get('id') == obj_id), None)
        if obj_post:
            obs.extend([
                1.0,
                (obj_post['x'] - ship['x']) / max(1, self.map_width),
                (obj_post['y'] - ship['y']) / max(1, self.map_height),
            ])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # Nearest wreckage (3 values): present + direction
        nearest_wreck = None
        if getattr(self, 'wreckage', None):
            nearest_wreck = min(
                self.wreckage,
                key=lambda w: (w['x'] - ship['x']) ** 2 + (w['y'] - ship['y']) ** 2,
            )
        if nearest_wreck:
            obs.extend([
                1.0,
                (nearest_wreck['x'] - ship['x']) / max(1, self.map_width),
                (nearest_wreck['y'] - ship['y']) / max(1, self.map_height),
            ])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # === ACTION RESTRICTIONS (38 values: 19 actions * 2 flags) ===
        # Encodes the active metadata.actionRestrictions matrix so the policy can
        # adapt when restrictions change (e.g., randomized per-episode). Aligned to
        # the 19-action mask order: [allowedWhileRecharging, allowedWithShieldsUp].
        obs.extend(self._action_restriction_features())

        # Return Dict observation with action mask
        obs_array = np.array(obs, dtype=np.float32)
        if skip_mask:
            mask = np.ones(self.num_action_types, dtype=np.int8)
        else:
            mask = self._get_action_mask(ship)
        return {
            'observation': obs_array,
            'action_mask': mask
        }

    def _action_restriction_features(self) -> List[float]:
        """Encode the active action-restriction matrix as 2 flags per action id.

        For every action id 0..num_action_types-1 (mask order), append
        `[allowedWhileRecharging, allowedWithShieldsUp]` from the action's
        `config['action_restrictions']` rule (defaulting to 1.0/allowed when the
        rule is absent). Reuses :data:`action_masker.ACTION_RESTRICTION_NAME` so the
        encoding stays in sync with the masker's gate.
        """
        restrictions = self.config.get('action_restrictions', {})
        feats: List[float] = []
        for action_id in range(self.num_action_types):
            key = action_masker.ACTION_RESTRICTION_NAME.get(action_id)
            rule = restrictions.get(key, {}) if key is not None else {}
            feats.append(1.0 if rule.get('allowedWhileRecharging', True) else 0.0)
            feats.append(1.0 if rule.get('allowedWithShieldsUp', True) else 0.0)
        return feats

    def _get_top_asteroids(self, x: int, y: int, count: int = 5) -> List[dict]:
        """Get top N asteroids ranked by a score combining mass, nutrinium concentration, and distance.

        Score formula: (nutrinium / mass) * nutrinium / (distance + 1)
        Higher score = better asteroid to target
        """
        if not self.asteroids:
            return []

        max_dist = math.sqrt(self.map_width**2 + self.map_height**2)
        scored_asteroids = []

        for asteroid in self.asteroids:
            if asteroid.get('nutrinium', 0) <= 0:
                continue

            dist = self._calculate_distance(x, y, asteroid['x'], asteroid['y'])
            mass = max(1, asteroid.get('mass', 1))
            nutrinium = asteroid.get('nutrinium', 0)

            # Calculate concentration (nutrinium / mass)
            concentration = nutrinium / mass

            # Score: concentration * nutrinium / (distance + 1)
            # This prioritizes: high concentration, high nutrinium, low distance
            raw_score = concentration * nutrinium / (dist + 1)

            # Normalize score to 0-1 range (approximate max score)
    max_score = 50.0 # Reasonable max for normalization
    normalized_score = min(1.0, raw_score / max_score)

    scored_asteroids.append({
        'x': asteroid['x'],
        'y': asteroid['y'],
        'mass': asteroid['mass'],
        'nutrinium': asteroid['nutrinium'],
        'distance': dist,
        'score': normalized_score,
    })

    # Sort by score descending and return top N
    scored_asteroids.sort(key=lambda a: a['score'], reverse=True)
    return scored_asteroids[:count]

    def _get_extreme_enemies(self, x: int, y: int) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Get the strongest and weakest active enemies.

        Strongest: highest combined health, energy, credits, and combat abilities
        Weakest: lowest combined values

        Returns: (strongest_enemy, weakest_enemy) - both at same coordinates as player for observation
        """
        active_enemies = [s for s in self.opponent_ships if not s.get('destroyed', False)]

        if not active_enemies:
            return None, None

        if len(active_enemies) == 1:
            return active_enemies[0], active_enemies[0]

        # Score each enemy
        scored_enemies = []
        for enemy in active_enemies:
            score = self._calculate_enemy_combat_score(enemy, raw=True)
            scored_enemies.append((score, enemy))

        # Sort by score
        scored_enemies.sort(key=lambda x: x[0], reverse=True)

        strongest = scored_enemies[0][1]
        weakest = scored_enemies[-1][1]

        return strongest, weakest

    def _calculate_enemy_combat_score(self, enemy: dict, raw: bool = False) -> float:
        """
        Calculate a combat score for an enemy ship.

        Factors: health, energy, credits, attack_power, attack_accuracy, shield_strength

        Args:
            enemy: Enemy ship dictionary
            raw: If True, return raw score; otherwise return normalized 0-1 score
        """
        health = enemy.get('health', 0)
        energy = enemy.get('energy', 0)
        credits = enemy.get('credits', 0)
        abilities = enemy.get('abilities', {})

        attack_power = abilities.get('attack_power', 0)
        attack_accuracy = abilities.get('attack_accuracy', 0)
        shield_strength = abilities.get('shield_strength', 0)
        evade = abilities.get('evade', 0)

        # Weighted score
        raw_score = (
            health * 1.0 +
            energy * 0.5 +
            credits * 0.1 +
            attack_power * 10.0 +
            attack_accuracy * 5.0 +
            shield_strength * 8.0 +
            evade * 3.0
        )

        if raw:
            return raw_score

        # Normalize (approximate max score)
        max_score = 100 + 50 + 100 + 100 + 50 + 80 + 30  # ~510
        return min(1.0, raw_score / max_score)

    def _get_info(self) -> dict:
        """
        Get additional information about the current state"""
        return {
            'step': self.current_step,
            'action_counter': self.action_counter,  # Track actions taken this episode
            'player_credits': self.player_ship['credits'],
            'player_nutrinium': self.player_ship['nutrinium'],
            'player_energy': self.player_ship['energy'],
            'player_health': self.player_ship['health'],
            'player_destroyed': self.player_ship['destroyed'],
            'asteroids_remaining': len([a for a in self.asteroids if a['nutrinium'] > 0]),
            'opponents_alive': len([s for s in self.opponent_ships if not s['destroyed']]),
            'invalid_action_count': getattr(self, 'invalid_action_count', 0),
            'state_invalid_action_count': getattr(self, 'state_invalid_action_count', 0),
        }