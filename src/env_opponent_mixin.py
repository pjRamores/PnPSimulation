"""Opponent / AI dispatch mixin for :class:`ProspectorsPiratesEnv`.

Holds opponent action selection: the dispatch table, the model-driven AI, the
delegating bot loaders (bot_v2..v6) plus their request composer / response
translators, enemy action-mask enforcement, and the enemy observation swap.
"""

from env_common import *


class EnvOpponentMixin:
    """Opponent action selection and built-in / delegating AIs."""

    def _get_opponent_action(self, ship: dict) -> int:
        """Dispatch opponent action based on assigned AI type"""
        ai_type = ship.get('ai_type', OpponentAIType.BOT_V2)

        if ai_type == OpponentAIType.MODEL:
            return self._ai_model(ship)
        elif ai_type == OpponentAIType.BOT_V3:
            return self._ai_bot_v3(ship)
        elif ai_type == OpponentAIType.BOT_V4:
            return self._ai_bot_v4(ship)
        elif ai_type == OpponentAIType.BOT_V5:
            return self._ai_bot_v5(ship)
        elif ai_type == OpponentAIType.BOT_V6:
            return self._ai_bot_v6(ship)
        elif ai_type == OpponentAIType.BOT_V7:
            return self._ai_bot_v7(ship)
        elif ai_type == OpponentAIType.BOT_V8:
            return self._ai_bot_v8(ship)
        else:  # BOT_V2 (default)
            return self._ai_bot_v2(ship)

    def _ai_model(self, ship: dict) -> int:
        """Model-based AI: Uses a trained RL model for decision-making.

        Args:
            ship: Enemy ship dictionary

        Returns:
            Action selected by the model (validated with action masking)
        """
        model_path = ship.get('model_path')

        if not model_path:
            logger.warning(f"MODEL AI type but no model_path specified for ship {ship.get('name')}. Falling back to BOT_V2.")
            return self._ai_bot_v2(ship)

        # Load model (uses cache if already loaded)
        model = self._load_enemy_model(model_path)

        if model is None:
            # _load_enemy_model() now emits one-time warnings and caches unavailable paths.
            # Avoid repeating warning spam each step when the model path is invalid.
            if model_path not in self._enemy_model_unavailable_paths:
                logger.warning(f"Failed to load model {model_path} for ship {ship.get('name')}. Falling back to BOT_V2.")
            return self._ai_bot_v2(ship)

        try:
            # Get observation from the enemy's perspective
            # We need to create an observation as if this enemy was the player
            # For simplicity, we'll use the same observation space but from enemy's viewpoint
            obs = self._get_enemy_observation(ship)

            # Predict action through compatibility adapter.
            # The adapter applies model-specific observation construction logic
            # (Dict vs flat, truncation) and action masking/remapping.
            action, _ = model.predict(obs, deterministic=True)

            # Convert to int (handle numpy arrays)
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.size == 1 else int(action[0])
            else:
                action = int(action)

            # Validate and enforce action masking for MODEL enemies
            # Without this, models waste turns on invalid actions (ATTACK with no enemy,
            # SELL with no trading post, MINE with no asteroid, etc.)
            action = self._enforce_enemy_action_mask(ship, action)

            return action

        except Exception as e:
            logger.warning(f"Error using model for ship {ship.get('name')}: {e}. Falling back to BOT_V2.")
            return self._ai_bot_v2(ship)

    def _get_bot_v2_module(self):
        """Lazily import the production bot (bot_v2) used by BOT_V2 opponents.

        bot_v2 lives in the sibling ``r680329-pnp-lambda`` folder and is pure
        stdlib, so it is imported by adding that directory to ``sys.path``. Its
        per-game file logging is disabled (PNP_DISABLE_LOGGING) so opponents
        never spawn log files / writer threads during training. The resolved
        module (or None when unavailable) is cached on the instance.
        """
        cached = getattr(self, '_bot_v2_module', 'unset')
        if cached != 'unset':
            return cached

        module = None
        try:
            import sys
            bot_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'bots'
            )
            if bot_dir not in sys.path:
                sys.path.insert(0, bot_dir)
            os.environ.setdefault('PNP_DISABLE_LOGGING', '1')
            import bot_v2
            if hasattr(bot_v2, '_refresh_logging_enabled'):
                bot_v2._refresh_logging_enabled()
            # Align the bot's coordinate frame with this environment. The env
            # uses the live server axis (N=y+1, S=y-1, E=x+1, W=x-1) -- exactly
            # the frame bot_v2 self-calibrates to in production. Pin its axis to
            # that frame so its compass directions map DIRECTLY onto the env's
            # MOVE actions (no inversion) and it never has to spend a move
            # learning the orientation. ``_axis`` is a module-level global the
            # bot exposes for this purpose.
            if hasattr(bot_v2, '_axis'):
                bot_v2._axis['ns'] = 1
                bot_v2._axis['ew'] = 1
            module = bot_v2
        except Exception as e:
            logger.warning(
                f"bot_v2 unavailable for BOT_V2 opponent AI: {e}. Falling back to HEURISTIC."
            )
            module = None

        self._bot_v2_module = module
        return module

    def _get_bot_v3_module(self):
        """Lazily import the prospector-economy bot (bot_v3) used by BOT_V3 opponents.

        Like bot_v2, bot_v3 lives in the sibling ``r680329-pnp-lambda`` folder
        and is pure stdlib. It emits MOVE directions in this environment's
        (live server) frame, so no axis pinning is needed. The resolved module
        (or None when unavailable) is cached on the instance.
        """
        cached = getattr(self, '_bot_v3_module', 'unset')
        if cached != 'unset':
            return cached

        module = None
        try:
            import sys
            bot_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'bots'
            )
            if bot_dir not in sys.path:
                sys.path.insert(0, bot_dir)
            import bot_v3
            module = bot_v3
        except Exception as e:
            logger.warning(
                f"bot_v3 unavailable for BOT_V3 opponent AI: {e}. Falling back to HEURISTIC."
            )
            module = None

        self._bot_v3_module = module
        return module

    def _get_bot_v4_module(self):
        """Lazily import the pirate-raider bot (bot_v4) used by BOT_V4 opponents.

        Like bot_v2/bot_v3, bot_v4 lives in the sibling ``r680329-pnp-lambda``
        folder and is pure stdlib. It emits MOVE directions in this
        environment's (live server) frame, so no axis pinning is needed. The
        resolved module (or None when unavailable) is cached on the instance.
        """
        cached = getattr(self, '_bot_v4_module', 'unset')
        if cached != 'unset':
            return cached

        module = None
        try:
            import sys
            bot_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'bots'
            )
            if bot_dir not in sys.path:
                sys.path.insert(0, bot_dir)
            import bot_v4
            module = bot_v4
        except Exception as e:
            logger.warning(
                f"bot_v4 unavailable for BOT_V4 opponent AI: {e}. Falling back to HEURISTIC."
            )
            module = None

        self._bot_v4_module = module
        return module

    def _get_bot_v5_module(self):
        """Lazily import the balanced miner-trader bot (bot_v5) used by BOT_V5 opponents.

        Like bot_v2/bot_v3/bot_v4, bot_v5 lives in the sibling
        ``r680329-pnp-lambda`` folder and is pure stdlib. It emits MOVE
        directions in this environment's (live server) frame, so no axis
        pinning is needed. The resolved module (or None when unavailable) is
        cached on the instance.
        """
        cached = getattr(self, '_bot_v5_module', 'unset')
        if cached != 'unset':
            return cached

        module = None
        try:
            import sys
            bot_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'bots'
            )
            if bot_dir not in sys.path:
                sys.path.insert(0, bot_dir)
            import bot_v5
            module = bot_v5
        except Exception as e:
            logger.warning(
                f"bot_v5 unavailable for BOT_V5 opponent AI: {e}. Falling back to HEURISTIC."
            )
            module = None

        self._bot_v5_module = module
        return module

    def _get_bot_v6_module(self):
        """Lazily import the model-backed bot (bot_v6) used by BOT_V6 opponents.

        Like the other delegating bots, bot_v6 lives in the sibling
        ``r680329-pnp-lambda`` folder; unlike them it is NOT pure stdlib (it
        loads a trained model via numpy / stable_baselines3) and reconstructs
        the observation from the ActionRequest. It emits MOVE directions in this
        environment's (live server) frame, so no axis pinning is needed. The
        resolved module (or None when unavailable) is cached on the instance.
        """
        cached = getattr(self, '_bot_v6_module', 'unset')
        if cached != 'unset':
            return cached

        module = None
        try:
            import sys
            bot_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'bots'
            )
            if bot_dir not in sys.path:
                sys.path.insert(0, bot_dir)
            import bot_v6
            module = bot_v6
        except Exception as e:
            logger.warning(
                f"bot_v6 unavailable for BOT_V6 opponent AI: {e}. Falling back to HEURISTIC."
            )
            module = None

        self._bot_v6_module = module
        return module

    def _get_bot_v7_module(self):
        """Lazily import the dummy-miner bot (bot_v7) used by BOT_V7 opponents.

        Like bot_v2..bot_v5, bot_v7 is pure stdlib and lives in the sibling
        ``bots`` folder. It emits MOVE directions in this environment's (live
        server) frame, so no axis pinning is needed. The resolved module (or
        None when unavailable) is cached on the instance.
        """
        cached = getattr(self, '_bot_v7_module', 'unset')
        if cached != 'unset':
            return cached

        module = None
        try:
            import sys
            bot_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'bots'
            )
            if bot_dir not in sys.path:
                sys.path.insert(0, bot_dir)
            import bot_v7
            module = bot_v7
        except Exception as e:
            logger.warning(
                f"bot_v7 unavailable for BOT_V7 opponent AI: {e}. Falling back to HEURISTIC."
            )
            module = None

        self._bot_v7_module = module
        return module

    def _get_bot_v8_module(self):
        """Lazily import the legacy model-backed bot (bot_v8) used by BOT_V8 opponents.

        Like bot_v6, bot_v8 is model-backed (it loads a trained model via numpy /
        stable_baselines3 and reconstructs the observation from the
        ActionRequest); unlike bot_v6 it wraps the legacy 128-dim / Discrete(14)
        v65 model. It already axis-corrects the v1 training frame internally and
        emits MOVE directions in this environment's (live server) frame, so no
        axis pinning is needed here. The resolved module (or None when
        unavailable) is cached on the instance.
        """
        cached = getattr(self, '_bot_v8_module', 'unset')
        if cached != 'unset':
            return cached

        module = None
        try:
            import sys
            bot_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'bots'
            )
            if bot_dir not in sys.path:
                sys.path.insert(0, bot_dir)
            import bot_v8
            module = bot_v8
        except Exception as e:
            logger.warning(
                f"bot_v8 unavailable for BOT_V8 opponent AI: {e}. Falling back to HEURISTIC."
            )
            module = None

        self._bot_v8_module = module
        return module

    def _compose_action_request(self, ship: dict) -> dict:
        """Compose a full ActionRequest dict for ``ship`` for the delegating bots.

        Shared by all bot opponents (BOT_V2/BOT_V3/BOT_V4/BOT_V5) -- the schema
        mirrors the live-server ActionRequest the bots parse in production (see
        ``logs/action_request.json``): ``actionId``, ``actionResult``,
        ``eventLog``, ``gameState`` (with full ``metadata``), ``leaderboard``,
        ``me`` and ``sensors``. The controlled ship becomes ``me``; sensor
        contacts become ``sensors``. Trading posts are ALWAYS included in full
        (every post on the map, regardless of sensor range), mirroring the
        production server which sends every trading post each tick as a known
        navigation landmark. Asteroids, enemy ships and wreckage remain
        sensor-limited (only those within the detection window are emitted) so
        exploration is still required. Environment config becomes
        ``gameState.metadata``. Fields the simulation does not track
        (transport/cosmetic: ``hue``, ``imageId``, ``requestConfig``,
        ``roundScores``, ``stats``) are emitted with neutral defaults so the
        shape matches production while the bots' consumed fields stay intact.
        """
        sensor_range = self.config['sensor_range']
        # The sensor_range ability widens the entity-DETECTION window only (more
        # candidates feed the fixed-size obs summary segments). The size-defining
        # local grid and metadata.sensors.range stay at the BASE range so the
        # reconstructed observation dimension never changes. Mirrors production,
        # where effective detection range = base sensors.range + sensor_range skill.
        _sensor_skill = int((ship.get('abilities') or {}).get('sensor_range', 0) or 0)
        effective_range = sensor_range + max(0, _sensor_skill)
        sx, sy = ship['x'], ship['y']
        game_id = id(ship)
        tick = getattr(self, 'current_step', 0)
        round_no = getattr(self, 'current_round', 0)

        def in_range(ex, ey):
            return max(abs(ex - sx), abs(ey - sy)) <= effective_range

        def ship_view(s: dict) -> dict:
            """Server-shaped view of a ship (used for ``me`` and ship sensors)."""
            return {
                '_id': str(s.get('id', '')),
                'credits': int(s.get('credits', 0)),
                'energy': int(s.get('energy', 0)),
                'gameId': game_id,
                'health': int(s.get('health', 0)),
                'hue': int(s.get('hue', 0)),
                'imageId': int(s.get('image_id', 0)),
                'location': {'x': s['x'], 'y': s['y']},
                'modules': list(s.get('modules', []) or []),
                'name': s.get('name'),
                'nutrinium': int(s.get('nutrinium', 0)),
                'objectives': dict(s.get('objectives', {}) or {}),
                'playerId': s.get('id'),
                'recharging': bool(s.get('recharging', False)),
                'roundScores': list(s.get('round_scores', []) or []),
                'shield': dict(s.get('shield', {}) or {}),
                'skillPointsSpent': int(s.get('skill_points_spent', 0)),
                'skillPointsTotal': int(s.get('skill_points_total', 0)),
                'skills': dict(s.get('abilities', {}) or {}),
                'state': s.get('state', 'READY'),
                'stats': dict(s.get('stats', {}) or {}),
                'teamId': s.get('team_id'),
                'type': 'ship',
            }

        sensors = []
        # Asteroids are static within an episode but sensor-limited; query only the
        # cells inside the sensor window (O(window)) instead of scanning every
        # entity (O(all)). _entities_in_window returns them in original list order
        # so the emitted sensors list is identical to the legacy full scan.
        for a in self._entities_in_window('asteroids', sx, sy, effective_range):
            sensors.append({
                'gameId': game_id,
                'location': {'x': a['x'], 'y': a['y']},
                'mass': int(a.get('mass', 0)),
                'name': a.get('name'),
                'nutrinium': int(a.get('nutrinium', 0)),
                'round': round_no,
                'type': 'asteroid',
            })
        # Trading posts are ALWAYS emitted in full (every post on the map),
        # regardless of sensor range, matching the production server which always
        # reports every trading post as a known navigation landmark. They are not
        # windowed like asteroids/enemies, so the policy can navigate to a post it
        # has not physically explored.
        for tp in self.trading_posts:
            sensors.append({
                'gameId': game_id,
                'id': tp.get('id'),
                'location': {'x': tp['x'], 'y': tp['y']},
                'name': tp.get('name'),
                'round': round_no,
                'type': 'trading_post',
            })
        for w in self.wreckage:
            if in_range(w['x'], w['y']):
                sensors.append({
                    'gameId': game_id,
                    'location': {'x': w['x'], 'y': w['y']},
                    'name': w.get('name'),
                    'nutrinium': int(w.get('nutrinium', 0)),
                    'round': round_no,
                    'type': 'wreckage',
                })
        other_ships = [self.player_ship] + [
            o for o in self.opponent_ships if o is not ship
        ]
        for other in other_ships:
            if other is None or other.get('destroyed', False):
                continue
            if not in_range(other['x'], other['y']):
                continue
            sensors.append(ship_view(other))

        me = ship_view(ship)
        me['requestConfig'] = dict(ship.get('request_config', {}) or {})

        costs = self.config['energy_costs']
        mining = self.config['mining']
        combat = self.config.get('combat', {})
        negotiate = self.config.get('negotiate', {})
        salvage = self.config.get('salvage', {})
        market = self.config.get('market', {})
        insurance = self.config.get('team_insurance', {})
        sell_price = round(
            getattr(self, 'market_price', market.get('sell_nutrinium', 0)), 2
        )
        metadata = {
            'actionRestrictions': dict(self.config.get('action_restrictions', {})),
            'combat': {
                'attackShieldDamage': combat.get('attack_shield_damage', 1.5),
                'baseHitChance': combat.get('base_target_number', 0.5),
                'baseShieldCapacity': combat.get('base_shield_capacity', 100),
                'baseShieldResistance': combat.get('base_shield_resistance', 0.25),
                'rechargePenalty': combat.get('recharge_penalty', 0.2),
                'shieldRechargeRate': combat.get('shield_recharge_rate', 10),
            },
            'mapConfig': {
                'asteroidDensity': self.config.get('asteroid_density', 0.11),
                'height': self.map_height,
                'maxMass': self.config.get('asteroid_mass_max', 500),
                'minMass': self.config.get('asteroid_mass_min', 50),
                'maxNutriniumPercent': self.config.get('nutrinium_max_percent', 1.0),
                'minNutriniumPercent': self.config.get('nutrinium_min_percent', 0.08),
                'tradingPostCount': getattr(
                    self, 'trading_post_target',
                    self._compute_trading_post_target()
                ),
                'width': self.map_width,
            },
            'market': {
                'buy': {
                    'repair': market.get('repair', 50),
                    'ship': market.get('ship_cost', 100),
                },
                # Current nutrinium sell price (drives bot_v5's market-timing).
                # The env updates ``market_price`` as cargo is sold; fall back to
                # the configured base price before the first sale.
                'sell': {'nutrinium': sell_price},
            },
            'mining': {
                'baseSuccessChance': mining.get('base_success_chance', 0.5),
                'maxPayout': mining.get('max_payout', 10),
                'minPayout': mining.get('min_payout', 1),
                'payoutModifier': mining.get('payout_modifier', 0.01),
            },
            'negotiate': {
                'baseFailChance': negotiate.get('base_fail_chance', 0.2),
                'baseSuccessChance': negotiate.get('base_success_chance', 0.4),
                'minFailChance': negotiate.get('min_fail_chance', 0.05),
            },
            'salvage': {
                'enabled': salvage.get('enabled', False),
                'energyCost': salvage.get('energy_cost', 5),
                'wreckagePercent': salvage.get('wreckage_percent', 0.5),
            },
            'shipConfig': {
                'energyCosts': {
                    'attack': costs.get('attack', 1),
                    'jump': costs.get('jump', 1),
                    'jumpMinCost': costs.get('jump_min_cost', 75),
                    'mine': costs.get('mine', 10),
                    'move': costs.get('move', 0),
                    'negotiate': costs.get('negotiate', 5),
                    'plunder': costs.get('plunder', 5),
                    'sell': costs.get('sell', 0),
                    'shieldMaintenance': costs.get('shield_maintenance', 1),
                    'shields': costs.get('shields', 1),
                },
                'energyPerRecharge': self.config['energy_per_recharge'],
                'maxEnergy': self.config['max_energy'],
                'maxJumpDistance': self.config['max_jump_distance'],
                'sensors': {'range': sensor_range},
            },
            'teamInsurance': {
                'baseCostPerMember': insurance.get('base_cost_per_member', 10),
                'costEscalation': insurance.get('cost_escalation', 1.0),
            },
        }

        # Leaderboard across every live participant, ranked by accumulated
        # credits (the simulation's stand-in for the server's gameScore).
        leaderboard = []
        for s in [self.player_ship] + list(self.opponent_ships):
            if s is None:
                continue
            leaderboard.append({
                'gameScore': int(s.get('credits', 0)),
                'playerId': s.get('id'),
                'shipName': s.get('name'),
            })
        leaderboard.sort(key=lambda e: e['gameScore'], reverse=True)
        for position, entry in enumerate(leaderboard, start=1):
            entry['position'] = position

        return {
            'actionId': f"{game_id}-{tick}",
            'actionResult': {
                'actionType': ship.get('last_action_type', 'WAIT'),
                'outcome': ship.get('last_action_outcome', 'SUCCESS'),
                'playerId': ship.get('id'),
                'ship': me,
            },
            'eventLog': list(getattr(self, 'event_log', []) or []),
            'gameState': {
                # ``gameId`` is unique and stable per ship+episode (ship objects
                # are recreated each reset). The bot keys its self-calibration /
                # move-history globals on gameId; a per-ship id keeps interleaved
                # opponents from clobbering each other's (pinned) axis mid-decision.
                'gameId': game_id,
                'metadata': metadata,
                'round': round_no,
                'tick': tick,
            },
            'leaderboard': leaderboard,
            'me': me,
            'sensors': sensors,
        }

    def _translate_bot_v2_move(self, ship: dict, payload: dict) -> int:
        """Map a bot_v2 MOVE direction directly onto an env MOVE action.

        The bot runs in this environment's (live server) coordinate frame --
        its axis is pinned to N=y+1/E=x+1 in ``_get_bot_v2_module`` -- so its
        compass directions align one-to-one with the env's MOVE actions and
        need no inversion: N->MOVE_NORTH, S->MOVE_SOUTH, E->MOVE_EAST,
        W->MOVE_WEST.
        """
        direction = str(payload.get('direction', '')).upper()
        mapping = {
            'N': ActionType.MOVE_NORTH,
            'S': ActionType.MOVE_SOUTH,
            'E': ActionType.MOVE_EAST,
            'W': ActionType.MOVE_WEST,
        }
        if direction in mapping:
            return int(mapping[direction])
        return int(ActionType.WAIT)

    def _translate_bot_action(
        self, ship: dict, response: dict
    ) -> Tuple[int, Optional[Tuple[int, int]], Optional[int]]:
        """Translate a bot response dict into a structured ``(action, target, energy)``.

        The simulator processes EXACTLY what the bot asks for -- there is no
        auto-targeting. A JUMP is sent to the bot's own ``target_location``
        coordinate (mapped to the generic coordinate-jump action, whatever sits
        at that cell), and an ATTACK carries the bot's requested energy payload.
        When a request cannot be carried out (out of jump range, insufficient
        energy, no asteroid/cargo, etc.) the underlying action simply fails and
        the ship does nothing that tick.

        Returns ``(action_int, target_xy_or_None, energy_or_None)``.
        """
        response = response or {}
        action_type = str(response.get('actionType', 'WAIT')).upper()
        payload = response.get('payload', {}) or {}
        simple = {
            'WAIT': ActionType.WAIT,
            'MINE': ActionType.MINE,
            'RECHARGE': ActionType.RECHARGE,
            'RECHARGE_END': ActionType.RECHARGE_END,
            'SELL': ActionType.SELL,
            'RAISE_SHIELDS': ActionType.RAISE_SHIELDS,
            'LOWER_SHIELDS': ActionType.LOWER_SHIELDS,
            'RESPAWN': ActionType.RESPAWN,
            'PLUNDER': ActionType.PLUNDER,
            'NEGOTIATE': ActionType.NEGOTIATE,
            'SALVAGE': ActionType.SALVAGE,
            'REPAIR': ActionType.REPAIR,
        }
        if action_type == 'MOVE':
            return self._translate_bot_v2_move(ship, payload), None, None
        if action_type == 'JUMP':
            target = payload.get('target_location') or {}
            tx, ty = target.get('x'), target.get('y')
            if tx is None or ty is None:
                return int(ActionType.WAIT), None, None
            # Coordinate jump to the bot's chosen cell (asteroid OR post); no
            # nearest-post / best-asteroid substitution.
            return int(ActionType.JUMP_TO_ASTEROID), (int(tx), int(ty)), None
        if action_type == 'ATTACK':
            energy = payload.get('energy')
            try:
                energy = int(energy)
            except (TypeError, ValueError):
                energy = None
            return int(ActionType.ATTACK), None, energy
        if action_type in simple:
            return int(simple[action_type]), None, None
        return int(ActionType.WAIT), None, None

    def _ai_bot_v2(self, ship: dict) -> int:
        """BOT_V2 AI: delegate the decision to the production heuristic bot.

        Composes an ActionRequest from the environment state, calls
        ``bot_v2.get_action``, and translates the response into a structured
        action. The bot's own target/energy are honored verbatim (no
        auto-targeting) and stashed on the ship for the step loop to execute;
        an action that is invalid for the current state simply does nothing
        (the underlying action method no-ops). Returns WAIT only when the bot
        itself is unavailable or errors.
        """
        bot = self._get_bot_v2_module()
        if bot is None:
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return int(ActionType.WAIT)

        try:
            request = self._compose_action_request(ship)
            response = bot.get_action(request)
            action, target, energy = self._translate_bot_action(ship, response)
            ship['_pending_action_target'] = target
            ship['_pending_action_energy'] = energy
            return action
        except Exception as e:
            logger.warning(
                f"Error using bot_v2 for ship {ship.get('name')}: {e}. Returning WAIT."
            )
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return int(ActionType.WAIT)

    def _ai_bot_v3(self, ship: dict) -> int:
        """BOT_V3 AI: delegate the decision to the prospector-economy bot.

        Reuses the generic ActionRequest composer and structured response
        translator (the bot_v3 response schema matches bot_v2's). The bot's own
        target/energy are honored verbatim (no auto-targeting); an action that
        is invalid for the current state simply does nothing. Falls back to
        BOT_V2 only when the bot itself is unavailable or errors.
        """
        bot = self._get_bot_v3_module()
        if bot is None:
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

        try:
            request = self._compose_action_request(ship)
            response = bot.get_action(request)
            action, target, energy = self._translate_bot_action(ship, response)
            ship['_pending_action_target'] = target
            ship['_pending_action_energy'] = energy
            return action
        except Exception as e:
            logger.warning(
                f"Error using bot_v3 for ship {ship.get('name')}: {e}. Falling back to BOT_V2."
            )
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

    def _ai_bot_v4(self, ship: dict) -> int:
        """BOT_V4 AI: delegate the decision to the pirate-raider bot.

        Reuses the generic ActionRequest composer and structured response
        translator (the bot_v4 response schema matches bot_v2's). The bot's own
        target/energy are honored verbatim (no auto-targeting); an action that
        is invalid for the current state simply does nothing. Falls back to
        BOT_V2 only when the bot itself is unavailable or errors.
        """
        bot = self._get_bot_v4_module()
        if bot is None:
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

        try:
            request = self._compose_action_request(ship)
            response = bot.get_action(request)
            action, target, energy = self._translate_bot_action(ship, response)
            ship['_pending_action_target'] = target
            ship['_pending_action_energy'] = energy
            return action
        except Exception as e:
            logger.warning(
                f"Error using bot_v4 for ship {ship.get('name')}: {e}. Falling back to BOT_V2."
            )
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

    def _ai_bot_v5(self, ship: dict) -> int:
        """BOT_V5 AI: delegate the decision to the balanced miner-trader bot.

        Reuses the generic ActionRequest composer and structured response
        translator (the bot_v5 response schema matches bot_v2's). The bot's own
        target/energy are honored verbatim (no auto-targeting); an action that
        is invalid for the current state simply does nothing. Falls back to
        BOT_V2 only when the bot itself is unavailable or errors.
        """
        bot = self._get_bot_v5_module()
        if bot is None:
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

        try:
            request = self._compose_action_request(ship)
            response = bot.get_action(request)
            action, target, energy = self._translate_bot_action(ship, response)
            ship['_pending_action_target'] = target
            ship['_pending_action_energy'] = energy
            return action
        except Exception as e:
            logger.warning(
                f"Error using bot_v5 for ship {ship.get('name')}: {e}. Falling back to BOT_V2."
            )
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

    def _ai_bot_v6(self, ship: dict) -> int:
        """BOT_V6 AI: delegate the decision to the model-backed bot.

        Reuses the generic ActionRequest composer and structured response
        translator (the bot_v6 response schema matches bot_v2's). bot_v6 loads a
        trained model and reconstructs the observation from the request, so its
        choices are model-driven rather than heuristic. The bot's own
        target/energy are honored verbatim (no auto-targeting); an action that
        is invalid for the current state simply does nothing. Falls back to
        BOT_V2 only when the bot itself is unavailable or errors.
        """
        bot = self._get_bot_v6_module()
        if bot is None:
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

        try:
            request = self._compose_action_request(ship)
            response = bot.get_action(request)
            action, target, energy = self._translate_bot_action(ship, response)
            ship['_pending_action_target'] = target
            ship['_pending_action_energy'] = energy
            return action
        except Exception as e:
            logger.warning(
                f"Error using bot_v6 for ship {ship.get('name')}: {e}. Falling back to BOT_V2."
            )
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

    def _ai_bot_v7(self, ship: dict) -> int:
        """BOT_V7 AI: delegate the decision to the dummy-miner bot.

        Reuses the generic ActionRequest composer and structured response
        translator (the bot_v7 response schema matches bot_v2's). bot_v7 only
        ever mines, recharges or wanders (random but on-map), so its response is
        always a simple MINE/RECHARGE/RECHARGE_END/MOVE/WAIT. The bot's own
        target/energy are honored verbatim (no auto-targeting); an action that
        is invalid for the current state simply does nothing. Falls back to
        BOT_V2 only when the bot itself is unavailable or errors.
        """
        bot = self._get_bot_v7_module()
        if bot is None:
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

        try:
            request = self._compose_action_request(ship)
            response = bot.get_action(request)
            action, target, energy = self._translate_bot_action(ship, response)
            ship['_pending_action_target'] = target
            ship['_pending_action_energy'] = energy
            return action
        except Exception as e:
            logger.warning(
                f"Error using bot_v7 for ship {ship.get('name')}: {e}. Falling back to BOT_V2."
            )
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

    def _ai_bot_v8(self, ship: dict) -> int:
        """BOT_V8 AI: delegate the decision to the legacy model-backed bot.

        Reuses the generic ActionRequest composer and structured response
        translator (the bot_v8 response schema matches bot_v2's). bot_v8 wraps
        the legacy 128-dim / Discrete(14) v65 model: it reconstructs the legacy
        observation from the request, predicts a scalar action, axis-corrects the
        v1 training frame, and enforces the current action mask. The bot's own
        target/energy are honored verbatim (no auto-targeting); an action that is
        invalid for the current state simply does nothing. Falls back to BOT_V2
        only when the bot itself is unavailable or errors.
        """
        bot = self._get_bot_v8_module()
        if bot is None:
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

        try:
            request = self._compose_action_request(ship)
            response = bot.get_action(request)
            action, target, energy = self._translate_bot_action(ship, response)
            ship['_pending_action_target'] = target
            ship['_pending_action_energy'] = energy
            return action
        except Exception as e:
            logger.warning(
                f"Error using bot_v8 for ship {ship.get('name')}: {e}. Falling back to BOT_V2."
            )
            ship['_pending_action_target'] = None
            ship['_pending_action_energy'] = None
            return self._ai_bot_v2(ship)

    def _enforce_enemy_action_mask(self, ship: dict, action: int) -> int:
        """Enforce action masking for an enemy ship, replacing invalid actions with valid ones.

        This gives MODEL enemies the same action enforcement the player gets,
        preventing them from wasting turns on invalid actions.
        """
        is_valid, reason = self._is_action_valid_for_state(action, ship, is_player=False)
        if is_valid:
            return action

        # Invalid action - apply fallback logic similar to player enforcement
        if ship.get('recharging', False):
            if ship['energy'] >= self.config['max_energy']:
                return int(ActionType.RECHARGE_END)
            elif action not in (int(ActionType.WAIT), int(ActionType.RECHARGE_END)):
                return int(ActionType.RECHARGE_END)
            else:
                return int(ActionType.WAIT)
        elif ship.get('destroyed', False):
            return int(ActionType.RESPAWN)
        else:
            # Pick the best valid action from the action mask
            mask = self._get_action_mask(ship, is_player=False)
            if ship['energy'] <= self.config['energy_costs'].get('move', 5):
                preferred = [
                    ActionType.RECHARGE, ActionType.MINE, ActionType.SELL,
                    ActionType.WAIT, ActionType.JUMP_TO_ASTEROID,
                    ActionType.JUMP_TO_TRADING_POST,
                    ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                    ActionType.MOVE_EAST, ActionType.MOVE_WEST,
                    ActionType.ATTACK, ActionType.RAISE_SHIELDS,
                ]
            else:
                preferred = [
                    ActionType.MINE, ActionType.SELL,
                    ActionType.JUMP_TO_ASTEROID, ActionType.JUMP_TO_TRADING_POST,
                    ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                    ActionType.MOVE_EAST, ActionType.MOVE_WEST,
                    ActionType.RECHARGE, ActionType.ATTACK,
                    ActionType.RAISE_SHIELDS, ActionType.WAIT,
                ]
            for fb in preferred:
                if mask[int(fb)] == 1:
                    return int(fb)
            return int(ActionType.WAIT)

    def _get_enemy_observation(self, enemy_ship: dict) -> Dict[str, np.ndarray]:
        """
        Get observation from an enemy ship's perspective.

        This creates an observation as if the enemy was the player,
        allowing us to use player-trained models for enemy AI.

        Args:
            enemy_ship: The enemy ship dictionary

        Returns:
            Dict observation compatible with the environment's observation space
        """
        # Use the enemy's assigned model spec to generate observation
        spec = self._get_ship_model_spec(enemy_ship)
        gen = self._get_observation_generator(spec.observation_spec)

        # Temporarily swap player and enemy to get enemy's perspective
        original_player = self.player_ship
        original_opponents = self.opponent_ships

        try:
            # Create a temporary opponent list (current player + other enemies, excluding this enemy)
            temp_opponents = [original_player] + [s for s in original_opponents if s != enemy_ship]

            # Temporarily set this enemy as the "player"
            self.player_ship = enemy_ship
            self.opponent_ships = temp_opponents

            # Generate observation using the assigned generator
            obs = gen.generate(enemy_ship)

            return obs

        finally:
            # Restore original state
            self.player_ship = original_player
            self.opponent_ships = original_opponents

