"""Spatial/geometry helper mixin for :class:`ProspectorsPiratesEnv`.

Distance math and entity-location lookups (including the per-step spatial
cache) used across observation, action and AI logic.
"""

from env_common import *


class EnvGeometryMixin:
    """Distance and entity-location helpers."""

    def _get_entity_at_location(self, x: int, y: int, entities: List[dict]) -> Optional[dict]:
        """Get entity at a specific location using spatial cache for performance."""
        # Try spatial cache first (keyed by list identity)
        cache = getattr(self, '_entity_location_cache', None)
        if cache is not None:
            # Use named keys for the canonical lists
            if entities is self.asteroids:
                loc_map = cache.get('asteroids')
            elif entities is self.trading_posts:
                loc_map = cache.get('trading_posts')
            elif entities is self.opponent_ships:
                loc_map = cache.get('opponents')
            else:
                loc_map = None
            if loc_map is not None:
                return loc_map.get((x, y))
        # Fallback: linear scan
        for entity in entities:
            if entity['x'] == x and entity['y'] == y:
                if entity.get('destroyed', False):
                    continue
                return entity
        return None

    def _build_static_location_cache(self):
        """Build (x,y)->entity and (x,y)->indices maps for the STATIC entity lists.

        Asteroid and trading-post positions do not change within an episode, so
        these maps are built once per reset and reused every step (only the moving
        opponents are refreshed by :meth:`_rebuild_location_cache`). The index map
        records every list position at each cell so windowed queries can return
        entities in original list order (see :meth:`_entities_in_window`).
        """
        static_loc = {}
        self._static_cell_index = {}
        for name, entities in (('asteroids', self.asteroids), ('trading_posts', self.trading_posts)):
            loc_map = {}
            idx_map = {}
            for idx, entity in enumerate(entities):
                if entity.get('destroyed', False):
                    continue
                key = (entity['x'], entity['y'])
                if key not in loc_map:
                    loc_map[key] = entity  # first wins (matches legacy linear scan)
                idx_map.setdefault(key, []).append(idx)
            static_loc[name] = loc_map
            self._static_cell_index[name] = idx_map
        self._static_location_cache = static_loc
        self._static_cache_ready = True

    def _rebuild_location_cache(self):
        """Refresh the per-step spatial lookup cache.

        The static lists (asteroids, trading posts) are cached once per reset by
        :meth:`_build_static_location_cache`; only the moving opponents are rebuilt
        here, so this is 0(opponents) per step instead of 0(all entities).
        """
        if not getattr(self, '_static_cache_ready', False):
            self._build_static_location_cache()
        opp_map = {}
        for entity in self.opponent_ships:
            if entity.get('destroyed', False):
                continue
            key = (entity['x'], entity['y'])
            if key not in opp_map:
                opp_map[key] = entity
        self._entity_location_cache = {
            'asteroids': self._static_location_cache['asteroids'],
            'trading_posts': self._static_location_cache['trading_posts'],
            'opponents': opp_map,
        }

    def _entities_in_window(self, name: str, sx: int, sy: int, r: int) -> List[dict]:
        """Return static entities within a Chebyshev window of radius ``r``.

        ``name`` is ``'asteroids'`` or ``'trading_posts'``. Uses the per-reset
        (x,y)->indices map to collect only entities whose cell lies in
        ``[sx-r, sx+r] x [sy-r, sy_r]``, returned in orginal list order so the
        result is identical to scanning the full list. Cst is O((2r+1)^2) instead
        of O(len(entities)) -- independent of map size.
        """
        idx_map = getattr(self, '_static_cell_index', {}).get(name)
        if idx_map is None:
            # Static cache not built yet (e.g. queried before the first step);
            # build it on demand so results stay correct.
            self._build_static_location_cache()
            idx_map = self._static_cell_index(name)
        if not idx_map:
            return []
        entities = self.asteroids if name == 'asteroids' else self.trading_posts
        found_indices: List[int] = []
        for x in range(sx - r, sx + r + 1):
            for y in range(sy - r, sy + r + 1):
                idxs = idx_map.get((x, y))
                if idxs:
                    found_indices.extend(idxs)
        found_indices.sort()
        return [entities[i] for i in found_indices]

    def _get_nearest_entity(self, x: int, y: int, entities: List[dict]) -> Optional[dict]:
        """Get nearest entity to a location"""
        if not entities:
            return None

        candidates = [
            e for e in entities
            if not ('destroyed' in e and e['destroyed'])
            and not ('nutrinium' in e and e['nutrinium'] <= 0)
        ]
        if not candidates:
            return None

        ex = np.fromiter((e['x'] for e in candidates), dtype=np.float64, count=len(candidates))
        ey = np.fromiter((e['y'] for e in candidates), dtype=np.float64, count=len(candidates))
        # Squared distance gives the same argmin as Euclidean distance; np.argmin
        # return the first occurrence, matching the legacy strict-less-than scan.
        d2 = (ex - x) ** 2 + (ey - y) ** 2
        return candidates[int(np.argmin(d2))]

    def _calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
