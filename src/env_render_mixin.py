"""
Rendering mixin for :class:`ProspectorsPiratesEnv`.

Contains the text/grid rendering helpers extracted from ``pnp_env`` to keep the core environment module focused on the Gym API surface.
"""

from env_common import *

class EnvRenderMixin:
    """Text/grid rendering helpers for the environment."""

    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            self._render_text()

    def _render_text(self):
        """Render the game state as text"""
        print(f'\n{'=' * 90}')
        print(f'Step: {self.current_step}/{self.max_steps}')
        print(f'{"=" * 90}')

        # Create map with wider spacing
        game_map = [[' ' for _ in range(self.map_width)] for _ in range(self.map_height)]

        # Place entities with more information
        # Store asteroid info for displaying nutrinium
        asteroid_map = {}
        for asteroid in self.asteroids:
            if asteroid['nutrinium'] > 0:
                x, y = asteroid['x'], asteroid['y']
                # Use numbers to represent asteroid size/nutrinium
                if asteroid['nutrinium'] > 40:
                    game_map[y][x] = '* '  # Large asteroid
                elif asteroid['nutrinium'] > 20:
                    game_map[y][x] = '* '  # Medium asteroid
                else:
                    game_map[y][x] = 'o '  # Small asteroid
                asteroid_map[(x, y)] = asteroid

        for post in self.trading_posts:
            game_map[post['y']][post['x']] = 'T '

        # Store enemy info
        enemy_map = {}
        for i, ship in enumerate(self.opponent_ships):
            if not ship['destroyed']:
                x, y = ship['x'], ship['y']
                game_map[y][x] = f"{i + 1} "  # Number enemies
                enemy_map[i] = ship

        if not self.player_ship['destroyed']:
            game_map[self.player_ship['y']][self.player_ship['x']] = 'P '

        # Print map with wider layout
        # Build a per-cell listing of entities so we can show multiple entities comma-separated
        cell_entities = [[[] for _ in range(self.map_width)] for _ in range(self.map_height)]

        # Asteroids: show as A<nutrinium> (only if nutrinium>0)
        for (x, y), asteroid in asteroid_map.items():
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                cell_entities[y][x].append(f"A{asteroid['nutrinium']}")

        # Trading posts
        for post in self.trading_posts:
            if 0 <= post['x'] < self.map_width and 0 <= post['y'] < self.map_height:
                cell_entities[post['y']][post['x']].append("T ")

        # Enemies - use ship names (E1, E2, etc.)
        for i, ship in enumerate(self.opponent_ships):
            if not ship.get('destroyed', False):
                x, y = ship['x'], ship['y']
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    # Use ship name instead of index
                    ship_name = ship.get('name', f'E{i + 1}')
                    cell_entities[y][x].append(ship_name)

        # Player - use ship name (P)
        if not self.player_ship.get('destroyed', False):
            px, py = self.player_ship['x'], self.player_ship['y']
            if 0 <= px < self.map_width and 0 <= py < self.map_height:
                # Use ship name for player
                player_name = self.player_ship.get('name', 'P')
                cell_entities[py][px].insert(0, player_name)

        # Prepare a bordered grid display. Choose a reasonable cell width.
        # Determine cell width (allow override via self.cell_width)
        if self.cell_width is not None and isinstance(self.cell_width, int) and self.cell_width > 0:
            cell_width = max(3, min(40, int(self.cell_width)))
        else:
            cell_width = max(6, min(12, 80 // max(1, self.map_width)))

        # Determine rendering window (full map or minimap around player)
        if self.minimap_mode and self.player_ship is not None:
            px, py = self.player_ship['x'], self.player_ship['y']
            r = max(0, int(self.minimap_radius))
            x_min = max(0, px - r)
            x_max = min(self.map_width - 1, px + r)
            y_min = max(0, py - r)
y_max = min(self.map_height - 1, py + r)
else:
    x_min, x_max = 0, self.map_width - 1
    y_min, y_max = 0, self.map_height - 1

x_count = x_max - x_min + 1

# Top header with column indices centered for the rendered window
header = '     ' + ''.join(str(i % 10).center(cell_width + 3) for i in range(x_min, x_max + 1))
print(header)

# Build box-drawing borders so each cell is enclosed (for the window width):
segment = '-' * (cell_width + 2)
top_border = '   ' + '+' + ''.join([segment] * x_count) + '+'
mid_border = '   ' + '+' + ''.join([segment] * x_count) + '+'
bottom_border = '   ' + '+' + ''.join([segment] * x_count) + '+'

print(top_border)

for y in range(y_min, y_max + 1):
    # Build row string with vertical separators
    row_cells = []
    for x in range(x_min, x_max + 1):
        items = cell_entities[y][x]
        if items:
            cell_text = ','.join(items)
        else:
            cell_text = ''
        # Truncate if too long
        if len(cell_text) > cell_width:
            cell_text = cell_text[:cell_width - 1] + '...'
        row_cells.append(cell_text.center(cell_width + 2))

    # Print row with left index and vertical separators
    print(f" {str(y % 10)} | " + ''.join(row_cells) + '|')

    # Print middle separator between rows (except after last rendered row)
    if y < y_max:
        print(mid_border)

# Bottom border
print(bottom_border)