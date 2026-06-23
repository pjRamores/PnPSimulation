"""
Spatial/geometry helper mixin for :class:`ProspectorsPiratesEnv`.

Distance math and entity-location lookups (including the per-step spatial cache) used across observation, action and AI logic.
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

    def _rebuild_location_cache(self):
        """Rebuild spatial lookup cache for all entity lists. Call once per step."""
        self._entity_location_cache = {}
        for name, entities in (('asteroids', self.asteroids), ('trading_posts', self.trading_posts), ('opponents', self.opponent_ships)):
            loc_map = {}
            for entity in entities:
                if entity.get('destroyed', False):
                    continue
                key = (entity['x'], entity['y'])
                if key not in loc_map:
                    loc_map[key] = entity
            self._entity_location_cache[name] = loc_map

    def _get_nearest_entity(self, x: int, y: int, entities: List[dict]) -> Optional[dict]:
        """Get nearest entity to a location"""
        if not entities:
            return None

        min_dist = float('inf')
        nearest = None

        for entity in entities:
            if 'destroyed' in entity and entity['destroyed']:
                continue
            if 'nutrinium' in entity and entity['nutrinium'] <= 0:
                continue

            dist = self._calculate_distance(x, y, entity['x'], entity['y'])
            if dist < min_dist:
                min_dist = dist
                nearest = entity

        return nearest

    def _calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)