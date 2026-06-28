"""Observation-generation mixin for :class:`ProspectorsPiratesEnv`.

Holds the per-ship ModelSpec plumbing and the legacy full-observation builder
along with the top-asteroid / extreme-enemy / combat-score helpers and the
episode info dict.
"""

from env_common import *
from utils import action_masker

# Map-size-invariant spatial encoding. Direction deltas and distances are
# normalized by a FIXED reference length (in cells) instead of the map
# dimensions, so identical relative geometry yields identical observation
# values on any map size. ~50 cells ≈ one default jump radius ("is the target
# within a jump?"). Absolute positions stay map-fraction (x / map_width) by
# design - "where am I on the map" is genuinely map-relative.
_SPATIAL_REF = 50.0


def _scaled_delta(d: float) -> float:
    """Scale a signed coordinate delta to [-1, 1] by a fixed reference length.

    Preserves direction (sign) and near-field magnitude; saturates for targets
    farther than ``_SPATIAL_REF`` cells. Map-size invariant.
    """
    return max(-1.0, min(1.0, d / _SPATIAL_REF))


def _scaled_distance(dist: float) -> float:
    """Scale a non-negative distance to [0, 1) via dist / (dist + ref).

    Smooth and monotonic (no hard clip), 0 at distance 0 and approaching 1 for
    large distances. Map-size invariant.
    """
    return dist / (dist + _SPATIAL_REF)


class EnvObservationMixin:
    """Observation construction and per-ship spec plumbing."""

    def _set_ship_model_spec(self, ship: dict, spec: Optional[ModelSpec] = None) -> None:
        """
        Associate a ModelSpec with a ship.

        Args:
            ship: Ship dictionary
            spec: ModelSpec instance (default = DEFAULT_FULL_SPEC)
        """
        if spec is None:
            spec = DEFAULT_FULL_SPEC
        self._ship_model_specs[id(ship)] = spec

    def _get_ship_model_spec(self, ship: dict) -> ModelSpec:
        """
        Get the ModelSpec for a ship, defaulting to full observation if not set.

        Args:
            ship: Ship dictionary

        Returns:
            ModelSpec instance
        """
        return self._ship_model_specs.get(id(ship), DEFAULT_FULL_SPEC)

    def _get_observation_generator(self, spec: ObservationSpec) -> ObservationGenerator:
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

    def _generate_observation_for_ship(self, ship: dict) -> Dict[str, np.ndarray]:
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

    def _get_partial_observation(self) -> Optional[Dict[str, np.ndarray]]:
        """Reconstruct the PLAYER's observation + action mask from a sensor-limited
        ActionRequest, using the same ``obs_reconstruction`` module BOT_V6 uses at
        inference. This gives byte-identical train/inference parity under partial
        observability. Returns ``None`` (caller falls back to the legacy/spec path)
        if the shared module is unavailable or reconstruction fails.
        """
        try:
            import obs_reconstruction
        except Exception:
            return None
        try:
            request = self._compose_action_request(self.player_ship)
            obs_vec = obs_reconstruction.build_observation(request, self.player_model_spec)
            mask = obs_reconstruction.build_action_mask(request)
            return {
                'observation': np.asarray(obs_vec, dtype=np.float32),
                'action_mask': np.asarray(mask, dtype=np.int8),
            }
        except Exception:
            return None

    def _get_observation(self, skip_mask: bool = False, use_spec: bool = True,
                         include_sensor_grid: bool = True) -> Dict[str, np.ndarray]:
        """Get the current observation with enhanced ship state and entity info.

        Args:
            skip_mask: If True, return a dummy action mask (all ones) to save computation.
                       Used for enemy observations where the mask is not needed.
            use_spec: If True, use the ship's assigned ModelSpec for observation generation.
                      If False, use the legacy full observation method.
            include_sensor_grid: If True (default), include the local sensor grid block
                       ((2*sensor_range+1)^2 values). If False, omit it entirely, producing
                       the FULL_NO_GRID layout (full observation minus the local sensor grid).
        """
        # Partial-observability mode: reconstruct the PLAYER observation + mask from
        # a sensor-limited ActionRequest via the shared module, byte-identical to what
        # a delegating BOT_V6 sees at inference. The recursion guard ensures inner
        # generators (which may call _get_observation) still use the legacy path.
        if getattr(self, 'partial_observability', False) and not getattr(self, '_generating_observation', False):
            partial = self._get_partial_observation()
            if partial is not None:
                return partial

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
        ])

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
        ast_here = self._get_entity_at_location(ship['x'], ship['y'], self.asteroids)
        obs.append(1.0 if (ast_here and ast_here.get('nutrinium', 0) > 0) else 0.0)

        # 2. At trading post?
        tp_here = self._get_entity_at_location(ship['x'], ship['y'], self.trading_posts)
        obs.append(1.0 if tp_here else 0.0)

        # 3. Cargo fullness (nutrinium as fraction of a "sell-worthy" amount ~25)
        obs.append(min(1.0, ship.get('nutrinium', 0) / 25.0))

        # 4. Enemy in same zone?
        enemy_here = any(
            e['x'] == ship['x'] and e['y'] == ship['y'] and not e.get('destroyed', False)
            for e in self.opponent_ships
        )
        obs.append(1.0 if enemy_here else 0.0)

        # 5-6. Direction to best asteroid (dx, dy, scale-free / map-size invariant)
        top_ast = self._get_top_asteroids(ship['x'], ship['y'], count=1)
        if top_ast:
            dx_ast = _scaled_delta(top_ast[0]['x'] - ship['x'])
            dy_ast = _scaled_delta(top_ast[0]['y'] - ship['y'])
        else:
            dx_ast, dy_ast = 0.0, 0.0
        obs.extend([dx_ast, dy_ast])

        # 7-8. Direction to nearest trading post (dx, dy, scale-free)
        nearest_tp = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
        if nearest_tp:
            dx_tp = _scaled_delta(nearest_tp['x'] - ship['x'])
            dy_tp = _scaled_delta(nearest_tp['y'] - ship['y'])
        else:
            dx_tp, dy_tp = 0.0, 0.0
        obs.extend([dx_tp, dy_tp])

        # === LOCAL SENSOR GRID (with clamped/shifted window to maximize valid cells) ===
        # Omitted entirely for the FULL_NO_GRID layout (include_sensor_grid=False).
        if include_sensor_grid:
            sensor_range = self.config['sensor_range']
            side = 2 * sensor_range + 1  # Grid dimension (e.g., 11 for sensor_range=5)

            # Calculate top-left corner of a centered window
            x_min = ship['x'] - sensor_range
            y_min = ship['y'] - sensor_range

            # Clamp window to stay within map bounds (shifts window when near edges)
            # This maximizes the number of valid cells in the observation
            x_min = max(0, min(x_min, self.map_width - side)) if self.map_width >= side else 0
            y_min = max(0, min(y_min, self.map_height - side)) if self.map_height >= side else 0

            if self.map_width >= side and self.map_height >= side:
                # Fast path: the clamped window is fully in bounds, so every cell is
                # valid (no -1.0 padding). Rather than scanning all side*side cells
                # (e.g. 441 lookups at sensor_range=10, almost all empty), start from
                # an all-empty grid and scatter only the occupied cells. Cost scales
                # with the number of nearby entities, not the grid area.
                grid = [0.0] * (side * side)
                cx = x_min + sensor_range  # window center matching [x_min, x_min+side-1]
                cy = y_min + sensor_range
                # Scatter in ascending priority so higher-priority entities overwrite
                # lower ones, matching the dense loop's enemy > post > asteroid elif.
                for a in self._entities_in_window('asteroids', cx, cy, sensor_range):
                    gx, gy = a['x'] - x_min, a['y'] - y_min
                    if 0 <= gx < side and 0 <= gy < side:
                        grid[gy * side + gx] = 0.33
                for p in self._entities_in_window('trading_posts', cx, cy, sensor_range):
                    gx, gy = p['x'] - x_min, p['y'] - y_min
                    if 0 <= gx < side and 0 <= gy < side:
                        grid[gy * side + gx] = 0.66
                # Enemies must come from the SAME per-step spatial cache the dense
                # path reads via _get_entity_at_location, so a stale cache yields
                # identical output (live opponent_ships could differ mid-step).
                cache = getattr(self, '_entity_location_cache', None)
                opp_map = cache.get('opponents') if cache is not None else None
                if opp_map is not None:
                    for (ex, ey), o in opp_map.items():
                        gx, gy = ex - x_min, ey - y_min
                        if 0 <= gx < side and 0 <= gy < side:
                            grid[gy * side + gx] = 1.0
                else:
                    for o in self.opponent_ships:
                        if o.get('destroyed', False):
                            continue
                        gx, gy = o['x'] - x_min, o['y'] - y_min
                        if 0 <= gx < side and 0 <= gy < side:
                            grid[gy * side + gx] = 1.0
                # Player's own cell is always reported empty.
                pgx, pgy = ship['x'] - x_min, ship['y'] - y_min
                if 0 <= pgx < side and 0 <= pgy < side:
                    grid[pgy * side + pgx] = 0.0
                obs.extend(grid)
            else:
                # Slow path: map smaller than the sensor grid, so some cells fall out
                # of bounds and must be padded with -1.0. Fill in row-major order.
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
                            elif self._get_entity_at_location(x, y, self.opponent_ships):
                                entity_type = 1.0
                            elif self._get_entity_at_location(x, y, self.trading_posts):
                                entity_type = 0.66
                            elif self._get_entity_at_location(x, y, self.asteroids):
                                entity_type = 0.33

                            obs.append(entity_type)
                        else:
                            # Out of bounds (should be rare with clamping, only when map < sensor grid)
                            obs.append(-1.0)

        # === TOP 5 ASTEROIDS (30 values: 5 asteroids * 6 features) ===
        top_asteroids = self._get_top_asteroids(ship['x'], ship['y'], count=self.config['top_asteroids_count'])
        max_mass = float(self.config.get('asteroid_mass_max', 80))

        for asteroid in top_asteroids:
            obs.extend([
                asteroid['x'] / max(1, self.map_width),
                asteroid['y'] / max(1, self.map_height),
                asteroid['mass'] / max(1.0, max_mass),
                asteroid['nutrinium'] / max(1.0, max_mass),
                _scaled_distance(asteroid['distance']),
                asteroid['score'],  # Already normalized 0-1
            ])

        # Pad with zeros if fewer than 5 asteroids
        for _ in range(self.config['top_asteroids_count'] - len(top_asteroids)):
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # === NEAREST TRADING POST (3 values) ===
        nearest_post = self._get_nearest_entity(ship['x'], ship['y'], self.trading_posts)
        if nearest_post:
            dist = self._calculate_distance(ship['x'], ship['y'], nearest_post['x'], nearest_post['y'])
            obs.extend([
                nearest_post['x'] / max(1, self.map_width),
                nearest_post['y'] / max(1, self.map_height),
                _scaled_distance(dist),
            ])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # === TWO ENEMY TYPES (16 values: 2 enemies * 8 features) ===
        # Get strongest and weakest enemies at same coordinates as player
        strongest, weakest = self._get_extreme_enemies(ship['x'], ship['y'])

        player_team = ship.get('team_id')
        player_team = int(player_team) if player_team is not None else 0
        for enemy in [strongest, weakest]:
            if enemy:
                combat_score = self._calculate_enemy_combat_score(enemy)
                enemy_team = enemy.get('team_id')
                same_team = 1.0 if (enemy_team is not None and int(enemy_team) == player_team) else 0.0
                obs.extend([
                    enemy['x'] / max(1, self.map_width),
                    enemy['y'] / max(1, self.map_height),
                    enemy['energy'] / max(1, self.config['max_energy']),
                    enemy['health'] / max(1, self.config['max_health']),
                    min(enemy['nutrinium'], 100) / 100.0,
                    min(enemy['credits'], 1000) / 1000.0,
                    combat_score,  # Already normalized 0-1
                    same_team,     # 1.0 if this enemy shares the player's team
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # === SPEC-FIDELITY FEATURES (24 values; appended to keep legacy offsets stable) ===
        max_abil = self.config.get('abilities', {})

        # New skills beyond the legacy 12-ability block (7 values)
        obs.extend([
            abilities.get('shield_capacity', 0) / max(1, max_abil.get('shield_capacity', 10)),
            abilities.get('shield_efficiency', 0) / max(1, max_abil.get('shield_efficiency', 10)),
            abilities.get('jump_cost', 0) / max(1, max_abil.get('jump_cost', 10)),
            abilities.get('salvage_yield', 0) / max(1, max_abil.get('salvage_yield', 10)),
            abilities.get('negotiate_skill', 0) / max(1, max_abil.get('negotiate_skill', 10)),
            abilities.get('negotiate_caution', 0) / max(1, max_abil.get('negotiate_caution', 10)),
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
        team_bonus = float(self.team_bonuses.get(team_id, 0.0)) if hasattr(self, 'team_bonuses') else 0.0
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
                _scaled_delta(obj_post['x'] - ship['x']),
                _scaled_delta(obj_post['y'] - ship['y']),
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
                _scaled_delta(nearest_wreck['x'] - ship['x']),
                _scaled_delta(nearest_wreck['y'] - ship['y']),
            ])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # === ACTION RESTRICTIONS (38 values: 19 actions * 2 flags) ===
        # Encodes the active metadata.actionRestrictions matrix so the policy can
        # adapt when restrictions change (e.g. randomized per-episode). Aligned to
        # the 19-action mask order: [allowedWhileRecharging, allowedWithShieldsUp].
        obs.extend(self._action_restriction_features())

        # === TEMPORAL/SPATIAL (2 values, appended last) ===
        # remaining_time_fraction: fraction of the game still left (1.0 at start ->
        #   0.0 at the end). Trains the policy to sell nutrinium before time runs
        #   out. Mirrors the action-counter normalization (self.action_counter /
        #   self.max_steps) so it stays byte-identical to the obs_reconstruction
        #   tick branch under partial-observability training.
        # quadrant_norm: the player's cell in a 3x3 map grid as a single normalized
        #   index q/8 (q = row*3 + col, 0..8). Encourages exploring other regions.
        remaining_time_fraction = max(
            0.0, min(1.0, (self.max_steps - self.action_counter) / max(1, self.max_steps))
        )
        col = min(2, (ship['x'] * 3) // max(1, self.map_width))
        row = min(2, (ship['y'] * 3) // max(1, self.map_height))
        quadrant_norm = (row * 3 + col) / 8.0
        obs.extend([remaining_time_fraction, quadrant_norm])

        # === PREY ENEMIES (9 values: 3 weakest huntable enemies * 3 features) ===
        # Appended after temporal/spatial so legacy offsets stay stable (old models
        # truncate this block). The top 3 weakest enemies that are NOT teammates,
        # have weaker attack AND defense than the player, are sensor-visible, and
        # hold nutrinium -- letting the policy hunt prey when no asteroid nutrinium
        # is available. Each prey: (x/W, y/H, nutrinium). Zero-padded if fewer.
        prey = self._get_prey_enemies(ship, count=3)
        for p in prey:
            obs.extend([
                p['x'] / max(1, self.map_width),
                p['y'] / max(1, self.map_height),
                min(p['nutrinium'], 100) / 100.0,
            ])
        for _ in range(3 - len(prey)):
            obs.extend([0.0, 0.0, 0.0])

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
        ``[allowedWhileRecharging, allowedWithShieldsUp]`` from the action's
        ``config['action_restrictions']`` rule (defaulting to 1.0/allowed when the
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


    def _get_top_asteroids(self, x: int, y: int, count: int = 5,
                           asteroids: Optional[List[dict]] = None) -> List[dict]:
        """
        Get top N asteroids ranked by a score combining mass, nutrinium concentration, and distance.

        Score formula: (nutrinium / mass) * nutrinium / (distance + 1)
        Higher score = better asteroid to target

        ``asteroids`` defaults to ``self.asteroids`` (the global, full-map list). Callers
        may pass a pre-filtered list (e.g. only sensor-visible asteroids) to rank a subset;
        the ranking is identical given the same ordered list.
        """
        source = self.asteroids if asteroids is None else asteroids
        if not source:
            return []

        # Live asteroids (nutrinium > 0) in list order so equal scores resolve the
        # same way as the legacy stable descending sort.
        live = [a for a in source if a.get('nutrinium', 0) > 0]
        if not live:
            return []

        n = len(live)
        ax = np.fromiter((a['x'] for a in live), dtype=np.float64, count=n)
        ay = np.fromiter((a['y'] for a in live), dtype=np.float64, count=n)
        mass = np.fromiter((max(1, a.get('mass', 1)) for a in live), dtype=np.float64, count=n)
        nutrinium = np.fromiter((a.get('nutrinium', 0) for a in live), dtype=np.float64, count=n)

        dist = np.sqrt((ax - x) ** 2 + (ay - y) ** 2)
        # Score: concentration * nutrinium / (distance + 1), normalized to 0-1.
        concentration = nutrinium / mass
        raw_score = concentration * nutrinium / (dist + 1.0)
        normalized_score = np.minimum(1.0, raw_score / 50.0)

        # Sort by score descending; stable keeps original list order for ties,
        # matching list.sort(key=..., reverse=True).3
        order = np.argsort(-normalized_score, kind='stable')[:count]

        result: List[dict] = []
        for i in order:
            a = live[int(i)]
            result.append({
                'x': a['x'],
                'y': a['y'],
                'mass': a['mass'],
                'nutrinium': a['nutrinium'],
                'distance': float(dist[i]),
                'score': float(normalized_score[i]),
            })
        return result

    def _visible_top_asteroids(self, ship: dict, count: int = 5) -> List[dict]:
        """Rank top-N asteroids for ``ship``, honoring partial observability.

        With ``partial_observability`` off (default), this is the global ranking over
        every asteroid on the map (legacy behaviour). With it on, only asteroids inside
        ``ship``'s sensor window are considered -- the same Chebyshev window
        (``config['sensor_range']`` widened by the ship's ``sensor_range`` skill) that
        :meth:`_compose_action_request` uses to build the sensor-limited ActionRequest.
        This keeps JUMP target resolution consistent with what the ship's (myopic)
        observation actually exposes, so action slot ``i`` points at obs asteroid ``i``.
        """
        if not getattr(self, 'partial_observability', False):
            return self._get_top_asteroids(ship['x'], ship['y'], count)

        sx, sy = ship['x'], ship['y']
        sensor_range = self.config['sensor_range']
        sensor_skill = int((ship.get('abilities') or {}).get('sensor_range', 0) or 0)
        effective_range = sensor_range + max(0, sensor_skill)
        # Filter preserving list order so the stable ranking matches the obs builder.
        visible = [
            a for a in self.asteroids
            if max(abs(a['x'] - sx), abs(a['y'] - sy)) <= effective_range
        ]
        return self._get_top_asteroids(sx, sy, count, asteroids=visible)

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

    def _get_prey_enemies(self, ship: dict, count: int = 3) -> List[dict]:
        """Return the top N weakest huntable enemies for the player to chase.

        An enemy qualifies as prey when it is, relative to ``ship``:
          - not destroyed and holding nutrinium (> 0),
          - not on the same team,
          - weaker in BOTH attack (attack_power + attack_accuracy) AND defense
            (shield_strength + evade), and
          - within the player's sensor range (Chebyshev distance <= sensor_range).
        Results are ranked weakest-first by the raw combat score and capped at
        ``count``. Each entry is ``{'x', 'y', 'nutrinium'}``.
        """
        sensor_range = self.config['sensor_range']
        px, py = ship['x'], ship['y']
        player_team = ship.get('team_id')
        player_team = int(player_team) if player_team is not None else 0

        pabil = ship.get('abilities', {}) or {}
        player_attack = pabil.get('attack_power', 0) + pabil.get('attack_accuracy', 0)
        player_defense = pabil.get('shield_strength', 0) + pabil.get('evade', 0)

        candidates = []
        for enemy in self.opponent_ships:
            if enemy.get('destroyed', False):
                continue
            if enemy.get('nutrinium', 0) <= 0:
                continue
            enemy_team = enemy.get('team_id')
            if enemy_team is not None and int(enemy_team) == player_team:
                continue
            # Sensor visibility (Chebyshev window matching the local sensor grid).
            if max(abs(enemy['x'] - px), abs(enemy['y'] - py)) > sensor_range:
                continue
            eabil = enemy.get('abilities', {}) or {}
            enemy_attack = eabil.get('attack_power', 0) + eabil.get('attack_accuracy', 0)
            enemy_defense = eabil.get('shield_strength', 0) + eabil.get('evade', 0)
            if not (enemy_attack < player_attack and enemy_defense < player_defense):
                continue
            score = self._calculate_enemy_combat_score(enemy, raw=True)
            candidates.append((score, enemy))

        # Weakest first.
        candidates.sort(key=lambda c: c[0])
        return [
            {'x': e['x'], 'y': e['y'], 'nutrinium': e.get('nutrinium', 0)}
            for _, e in candidates[:count]
        ]

    def _get_info(self) -> dict:
        """Get additional information about the current state"""
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
