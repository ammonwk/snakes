import pygame
import random
import sys
import numpy as np
from collections import deque
import os

# Game settings
CELL_SIZE = 20
VISIBLE_GRID_WIDTH, VISIBLE_GRID_HEIGHT = 30, 30  # Viewport size
WORLD_GRID_WIDTH, WORLD_GRID_HEIGHT = 150, 150  # Total world size
WIDTH, HEIGHT = VISIBLE_GRID_WIDTH * CELL_SIZE, VISIBLE_GRID_HEIGHT * CELL_SIZE
REPRODUCE_AT_LENGTH = 20  # Length at which snakes can reproduce

# Colors
FOOD_COLOR = (255, 100, 100)
BORDER_COLOR = (255, 255, 255)
BG_TOP_COLOR = (20, 20, 40)
BG_BOTTOM_COLOR = (0, 0, 20)

# Graph colors
GRAPH_BG = (30, 30, 50)
GRAPH_GRID = (60, 60, 80)
GRAPH_MEAN = (50, 130, 255)  # Blue line
GRAPH_STD = (50, 130, 255, 80)  # Semi-transparent blue
GRAPH_MINMAX = (150, 150, 150, 40)  # Semi-transparent gray
GRAPH_TEXT = (220, 220, 220)
GRAPH_TITLE = (255, 255, 255)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Genetic algorithm settings
MUTATION_RATE = 0.2
MUTATION_AMOUNT = 0.4


class GeneTracker:
    """Tracks genetic statistics across generations"""

    def __init__(self):
        # Dictionary to store historical data
        self.history = {"generation": [], "mean": {}, "std": {}, "min": {}, "max": {}}

        # List of gene names to track
        self.gene_names = [
            "SAFETY_WEIGHT",
            "AREA_WEIGHT",
            "FOOD_WEIGHT",
            "TRAP_WEIGHT",
            "FUTURE_OPTIONS_WEIGHT",
            "AGGRESSION",
            "FOOD_PURSUIT",
            "SURVIVAL_INSTINCT",
        ]

        # Initialize history for each gene
        for gene in self.gene_names:
            self.history["mean"][gene] = []
            self.history["std"][gene] = []
            self.history["min"][gene] = []
            self.history["max"][gene] = []

        # Current generation
        self.current_generation = 0

    def update_stats(self, snakes):
        """Update statistics based on current snake population"""
        # Skip if no snakes alive
        alive_snakes = [s for s in snakes if s.alive]
        if not alive_snakes:
            return

        # Get current max generation
        max_gen = max(snake.generation for snake in alive_snakes)

        # Only record if this is a new generation
        if max_gen > self.current_generation:
            self.current_generation = max_gen
            self.history["generation"].append(max_gen)

            # Collect all gene values
            gene_values = {gene: [] for gene in self.gene_names}

            for snake in alive_snakes:
                for gene in self.gene_names:
                    gene_values[gene].append(snake.ai.genes[gene])

            # Calculate statistics
            for gene in self.gene_names:
                values = gene_values[gene]
                if values:
                    self.history["mean"][gene].append(np.mean(values))
                    self.history["std"][gene].append(np.std(values))
                    self.history["min"][gene].append(min(values))
                    self.history["max"][gene].append(max(values))
                else:
                    # No data for this gene
                    self.history["mean"][gene].append(0)
                    self.history["std"][gene].append(0)
                    self.history["min"][gene].append(0)
                    self.history["max"][gene].append(0)

    def create_gene_graph(self, gene_name, width, height):
        """Create a pygame surface with a graph for a specific gene"""
        if not self.history["generation"]:
            # No data yet
            return pygame.Surface((width, height))

        # Create the surface
        surface = pygame.Surface((width, height))
        surface.fill(GRAPH_BG)

        # Set padding - INCREASED TOP PADDING
        padding_left = 60  # Space for y-axis labels
        padding_right = 20
        padding_top = 40  # Increased from 40 to 55 for title
        padding_bottom = 30  # Space for x-axis labels

        # Calculate plot area
        plot_width = width - padding_left - padding_right
        plot_height = height - padding_top - padding_bottom
        plot_rect = pygame.Rect(padding_left, padding_top, plot_width, plot_height)

        # Get data
        generations = self.history["generation"]
        mean_values = self.history["mean"][gene_name]
        std_values = self.history["std"][gene_name]
        min_values = self.history["min"][gene_name]
        max_values = self.history["max"][gene_name]

        # Find value range for y-axis scaling
        min_y = min(min_values) if min_values else 0
        max_y = max(max_values) if max_values else 1

        # Add a bit of padding to the range
        y_padding = (max_y - min_y) * 0.1
        min_y = max(0, min_y - y_padding)  # Ensure we don't go below 0
        max_y = max_y + y_padding

        # Draw grid
        pygame.draw.rect(surface, GRAPH_GRID, plot_rect, 1)

        # Draw horizontal grid lines
        for i in range(1, 5):
            y = padding_top + i * plot_height // 5
            pygame.draw.line(surface, GRAPH_GRID, (padding_left, y), (padding_left + plot_width, y))

            # Label y-axis
            value = max_y - i * (max_y - min_y) / 5
            font = pygame.font.SysFont("Arial", 12)
            text = font.render(f"{value:.1f}", True, GRAPH_TEXT)
            text_rect = text.get_rect(right=padding_left - 5, centery=y)
            surface.blit(text, text_rect)

        # Label y-axis top value
        y_top_text = font.render(f"{max_y:.1f}", True, GRAPH_TEXT)
        y_top_rect = y_top_text.get_rect(right=padding_left - 5, centery=padding_top)
        surface.blit(y_top_text, y_top_rect)

        # Draw vertical grid lines
        if len(generations) > 1:
            step = max(1, len(generations) // 5)
            for i in range(0, len(generations), step):
                # Scale to plot coordinates
                x = padding_left + i * plot_width // (len(generations) - 1)
                pygame.draw.line(
                    surface, GRAPH_GRID, (x, padding_top), (x, padding_top + plot_height)
                )

                # Label x-axis
                gen_text = font.render(f"{generations[i]}", True, GRAPH_TEXT)
                gen_rect = gen_text.get_rect(centerx=x, top=padding_top + plot_height + 5)
                surface.blit(gen_text, gen_rect)

        # Helper function to map data coordinates to screen coordinates
        def data_to_screen(gen_idx, value):
            x = padding_left
            if len(generations) > 1:
                x += gen_idx * plot_width // (len(generations) - 1)
            y = padding_top + plot_height - (value - min_y) * plot_height / (max_y - min_y)
            return int(x), int(y)

        # Draw min/max range (filled area)
        if len(generations) > 1:
            # Create polygon points for fill
            points = []
            for i in range(len(generations)):
                points.append(data_to_screen(i, max_values[i]))

            # Add points in reverse order for bottom line
            for i in range(len(generations) - 1, -1, -1):
                points.append(data_to_screen(i, min_values[i]))

            # Draw filled polygon
            if len(points) >= 3:  # Need at least 3 points for a polygon
                # Create a transparent surface for the polygon
                poly_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                minmax_color = (*GRAPH_MINMAX[:3], GRAPH_MINMAX[3])
                pygame.draw.polygon(poly_surface, minmax_color, points)
                surface.blit(poly_surface, (0, 0))

        # Draw standard deviation range (filled area)
        if len(generations) > 1:
            # Create polygon points for fill
            points = []
            for i in range(len(generations)):
                points.append(data_to_screen(i, mean_values[i] + std_values[i]))

            # Add points in reverse order for bottom line
            for i in range(len(generations) - 1, -1, -1):
                points.append(data_to_screen(i, mean_values[i] - std_values[i]))

            # Draw filled polygon
            if len(points) >= 3:  # Need at least 3 points for a polygon
                # Create a transparent surface for the polygon
                poly_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                std_color = (*GRAPH_STD[:3], GRAPH_STD[3])
                pygame.draw.polygon(poly_surface, std_color, points)
                surface.blit(poly_surface, (0, 0))

        # Draw mean line
        if len(generations) > 1:
            points = [data_to_screen(i, mean_values[i]) for i in range(len(generations))]

            # Draw line segments
            for i in range(len(points) - 1):
                pygame.draw.line(surface, GRAPH_MEAN, points[i], points[i + 1], 2)

        # Draw title
        title_font = pygame.font.SysFont("Arial", 16, bold=True)
        title_text = title_font.render(f"{gene_name} Evolution", True, GRAPH_TITLE)
        title_rect = title_text.get_rect(centerx=width // 2, top=10)  # More space from top
        surface.blit(title_text, title_rect)

        # Add current value annotation
        if mean_values:
            current_value = mean_values[-1]
            current_std = std_values[-1]

            value_font = pygame.font.SysFont("Arial", 14)
            value_text = value_font.render(
                f"Current: {current_value:.2f} ±{current_std:.2f}", True, (255, 255, 255)
            )
            value_bg = pygame.Rect(width - 160, 5, 155, 20)

            # Draw semi-transparent background for text
            bg_surf = pygame.Surface((155, 20), pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 128))
            surface.blit(bg_surf, (width - 160, 110))

            surface.blit(value_text, (width - 155, 110))

        return surface

    def create_all_graphs(self):
        """Create small graphs for all genes combined into one surface"""
        graph_width = WIDTH
        graph_height = HEIGHT

        # Create a surface to hold all graphs
        combined_surface = pygame.Surface((graph_width, graph_height))
        combined_surface.fill(GRAPH_BG)  # Dark background

        # Calculate grid layout
        cols = 2
        rows = len(self.gene_names) // cols
        if len(self.gene_names) % cols != 0:
            rows += 1

        cell_width = graph_width // cols
        cell_height = (graph_height - 20) // rows  # Subtract space for main title

        # Add main title with more space
        font = pygame.font.SysFont("Arial", 28, bold=True)
        title = font.render("Genetic Evolution Over Generations", True, (255, 255, 255))
        title_rect = title.get_rect(centerx=graph_width // 2, top=10)
        combined_surface.blit(title, title_rect)

        # Create each graph and position it in the grid
        for i, gene in enumerate(self.gene_names):
            row = i // cols
            col = i % cols

            # Create the graph
            graph = self.create_gene_graph(gene, cell_width - 10, cell_height - 10)

            # Position in the grid - START AT Y=50 to leave space for title
            x = col * cell_width + 10
            y = 50 + row * cell_height + 10

            # Draw a border
            border_rect = pygame.Rect(x, y, cell_width - 10, cell_height - 10)
            pygame.draw.rect(combined_surface, (100, 100, 100), border_rect, 2)

            # Blit the graph
            combined_surface.blit(graph, (x, y))

        # Add instructions
        instruction_font = pygame.font.SysFont("Arial", 16)
        instruction = instruction_font.render("Press G to close graphs", True, (200, 200, 200))
        instruction_rect = instruction.get_rect(bottomright=(graph_width - 10, graph_height - 10))
        combined_surface.blit(instruction, instruction_rect)

        return combined_surface


class SnakeAI:
    """Represents a snake's AI decision-making genes"""

    def __init__(self, parent=None):
        if parent:
            # Inherit genes with mutation
            self.genes = self._mutate_genes(parent.genes)
        else:
            # Generate random initial genes
            self.genes = {
                "SAFETY_WEIGHT": random.uniform(80, 120),
                "AREA_WEIGHT": random.uniform(1, 5),
                "FOOD_WEIGHT": random.uniform(20, 50),
                "TRAP_WEIGHT": random.uniform(5, 30),
                "FUTURE_OPTIONS_WEIGHT": random.uniform(10, 25),
                "AGGRESSION": random.uniform(0.5, 2.0),
                "FOOD_PURSUIT": random.uniform(0.5, 2.0),
                "SURVIVAL_INSTINCT": random.uniform(0.5, 2.0),
            }

        # Stats for evolution analysis
        self.age = 0
        self.food_eaten = 0
        self.kills = 0

    def _mutate_genes(self, parent_genes):
        """Create a mutated copy of parent genes"""
        new_genes = {}
        for gene, value in parent_genes.items():
            if random.random() < MUTATION_RATE:
                # Apply mutation
                mutation_factor = 1 + random.uniform(-MUTATION_AMOUNT, MUTATION_AMOUNT)
                new_genes[gene] = max(0.1, value * mutation_factor)  # Ensure no negative weights
            else:
                # Direct inheritance
                new_genes[gene] = value
        return new_genes

    def generate_colors(self):
        """Generate colors based on genetic traits"""
        # Map FOOD_PURSUIT to red (higher = redder)
        r = int(min(255, self.genes["FOOD_PURSUIT"] * 127))

        # Map AGGRESSION to green (higher = greener)
        g = int(min(255, self.genes["AGGRESSION"] * 127))

        # Map SURVIVAL_INSTINCT to blue (higher = bluer)
        b = int(min(255, self.genes["SURVIVAL_INSTINCT"] * 127))

        # Create head and body colors
        head_color = (r, g, b)
        body_color = (min(255, r + 30), min(255, g + 30), min(255, b + 30))

        return head_color, body_color


class Snake:
    """Represents a snake in the game"""

    next_id = 0

    def __init__(self, segments, direction, ai=None, parent_id=None):
        self.id = Snake.next_id
        Snake.next_id += 1

        self.segments = segments.copy()  # List of (x, y) positions
        self.direction = direction
        self.alive = True
        self.ai = ai if ai else SnakeAI()  # Create new AI if none provided
        self.parent_id = parent_id
        self.generation = 1 if parent_id is None else 0  # Will be updated for children

        # Generate colors based on genetics
        self.color_head, self.color_body = self.ai.generate_colors()

    def move(self, food, other_snakes):
        """Move the snake based on AI decision"""
        # Create obstacles set containing all snake segments
        obstacles = set()
        for other_snake in other_snakes:
            obstacles.update(other_snake.segments)

        # Add own segments except the tail (which will move)
        if len(self.segments) > 1:
            obstacles.update(self.segments[:-1])

        # Get move from AI
        self.direction = self.get_ai_move(food, obstacles, other_snakes)

        # Calculate new head position
        new_head = (
            self.segments[0][0] + self.direction[0],
            self.segments[0][1] + self.direction[1],
        )

        # Check if eating food
        eating = any(new_head == f for f in food)

        # Move snake
        self.segments.insert(0, new_head)
        if not eating:
            self.segments.pop()
        else:
            self.ai.food_eaten += 1
            return new_head  # Return the food position that was eaten

        return None  # No food eaten

    def check_collision(self, other_snakes):
        """Check if snake has collided with walls or other snakes"""
        head = self.segments[0]

        # Check wall collision
        if not (0 <= head[0] < WORLD_GRID_WIDTH and 0 <= head[1] < WORLD_GRID_HEIGHT):
            self.alive = False
            return

        # Check self collision
        if head in self.segments[1:]:
            self.alive = False
            return

        # Check collision with other snakes
        for other_snake in other_snakes:
            if head in other_snake.segments:
                self.alive = False
                # If we hit another snake's head, they die too
                if head == other_snake.segments[0]:
                    other_snake.alive = False
                else:
                    # Register a kill
                    other_snake.ai.kills += 1
                return

    def should_reproduce(self):
        """Check if snake should reproduce (length >= 10)"""
        return len(self.segments) >= REPRODUCE_AT_LENGTH

    def reproduce(self):
        """Split into two snakes of length 5 with mutated AI"""
        if len(self.segments) < REPRODUCE_AT_LENGTH:
            return []

        # First 5 segments go to first child
        child1_segments = self.segments[: REPRODUCE_AT_LENGTH // 2]

        # Last 5 segments go to second child
        child2_segments = self.segments[REPRODUCE_AT_LENGTH // 2 :]

        # Create children with mutated AI
        child1 = Snake(child1_segments, self.direction, SnakeAI(self.ai), self.id)

        # Second child moves in a perpendicular direction if possible
        child2_direction = self._get_perpendicular_direction()

        child2 = Snake(child2_segments, child2_direction, SnakeAI(self.ai), self.id)

        # Set generation
        child1.generation = self.generation + 1
        child2.generation = self.generation + 1

        return [child1, child2]

    def _get_perpendicular_direction(self):
        """Get a direction perpendicular to current direction"""
        if self.direction in [UP, DOWN]:
            return random.choice([LEFT, RIGHT])
        else:
            return random.choice([UP, DOWN])

    def get_ai_move(self, food, obstacles, other_snakes):
        """Modified AI snake movement with genetic parameters"""
        possible_directions = [UP, DOWN, LEFT, RIGHT]
        opposite = (-self.direction[0], -self.direction[1])
        possible_directions = [d for d in possible_directions if d != opposite]

        # Get the closest food
        closest_food = None
        min_distance = float("inf")
        for f in food:
            dist = abs(self.segments[0][0] - f[0]) + abs(self.segments[0][1] - f[1])
            if dist < min_distance:
                min_distance = dist
                closest_food = f

        if not closest_food:
            return self.direction  # No food, maintain direction

        # Find the nearest other snake head
        nearest_opponent_head = None
        min_opponent_dist = float("inf")

        # Find nearest opponent's head position
        for other_snake in other_snakes:
            other_head = other_snake.segments[0]
            dist = abs(self.segments[0][0] - other_head[0]) + abs(
                self.segments[0][1] - other_head[1]
            )
            if dist < min_opponent_dist:
                min_opponent_dist = dist
                nearest_opponent_head = other_head

        # Calculate if we're bigger than nearby opponents
        our_length = len(self.segments)
        is_bigger = False
        for other_snake in other_snakes:
            if len(other_snake.segments) + 2 < our_length:
                is_bigger = True
                break

        # Weights from genes
        SAFETY_WEIGHT = self.ai.genes["SAFETY_WEIGHT"]
        AREA_WEIGHT = self.ai.genes["AREA_WEIGHT"]
        FOOD_WEIGHT = self.ai.genes["FOOD_WEIGHT"] * self.ai.genes["FOOD_PURSUIT"]
        TRAP_WEIGHT = self.ai.genes["TRAP_WEIGHT"] * self.ai.genes["AGGRESSION"]
        FUTURE_OPTIONS_WEIGHT = (
            self.ai.genes["FUTURE_OPTIONS_WEIGHT"] * self.ai.genes["SURVIVAL_INSTINCT"]
        )

        # Evaluate each possible move
        best_direction = self.direction
        best_score = float("-inf")

        for d in possible_directions:
            new_head = (self.segments[0][0] + d[0], self.segments[0][1] + d[1])

            # Skip invalid moves
            if not (0 <= new_head[0] < WORLD_GRID_WIDTH and 0 <= new_head[1] < WORLD_GRID_HEIGHT):
                continue
            if new_head in obstacles:
                continue

            # Check if we'll eat food
            will_eat = new_head in food

            # Calculate available space
            available_area = self._flood_fill_count(new_head, obstacles)

            # Distance to food
            distance_to_food = abs(new_head[0] - closest_food[0]) + abs(
                new_head[1] - closest_food[1]
            )

            # Calculate immediate safety
            immediate_safety = False
            for future_d in [UP, DOWN, LEFT, RIGHT]:
                future_pos = (new_head[0] + future_d[0], new_head[1] + future_d[1])
                if (
                    0 <= future_pos[0] < WORLD_GRID_WIDTH
                    and 0 <= future_pos[1] < WORLD_GRID_HEIGHT
                    and future_pos not in obstacles
                ):
                    immediate_safety = True
                    break

            # Calculate opponent factors
            opponent_factor = 0
            if nearest_opponent_head:
                opponent_head_distance = abs(new_head[0] - nearest_opponent_head[0]) + abs(
                    new_head[1] - nearest_opponent_head[1]
                )
                # If we're bigger, being close to opponent is good for trapping
                if is_bigger:
                    opponent_factor = (
                        max(0, 5 - opponent_head_distance) * self.ai.genes["AGGRESSION"]
                    )
                # If we're smaller, keep distance unless going for food
                else:
                    opponent_factor = (
                        -max(0, 5 - opponent_head_distance) * self.ai.genes["SURVIVAL_INSTINCT"]
                    )

            # Calculate score for this move
            score = (
                (1000 if immediate_safety else -1000 * self.ai.genes["SURVIVAL_INSTINCT"])  # Safety
                + (available_area * AREA_WEIGHT)  # Space
                + (100 if will_eat else -distance_to_food * FOOD_WEIGHT)  # Food
                + opponent_factor  # Opponent factor
            )

            if score > best_score:
                best_score = score
                best_direction = d

        return best_direction

    def _flood_fill_count(self, start, obstacles):
        """Count accessible cells from a position"""
        visited = set()
        queue = deque([start])
        count = 0

        while queue and count < 100:  # Limit check to nearby area for performance
            cell = queue.popleft()
            if cell in visited:
                continue
            visited.add(cell)
            count += 1

            for d in [UP, DOWN, LEFT, RIGHT]:
                neighbor = (cell[0] + d[0], cell[1] + d[1])
                if (
                    0 <= neighbor[0] < WORLD_GRID_WIDTH
                    and 0 <= neighbor[1] < WORLD_GRID_HEIGHT
                    and neighbor not in obstacles
                    and neighbor not in visited
                ):
                    queue.append(neighbor)

        return count


class Camera:
    """Handles viewport scrolling"""

    def __init__(self):
        self.x = 0
        self.y = 0
        self.scale = 1  # Normal scale

    def center_on(self, pos):
        """Center the camera on a position"""
        self.x = pos[0] - VISIBLE_GRID_WIDTH // 2
        self.y = pos[1] - VISIBLE_GRID_HEIGHT // 2

        # Keep camera within world bounds
        self.x = max(0, min(WORLD_GRID_WIDTH - VISIBLE_GRID_WIDTH, self.x))
        self.y = max(0, min(WORLD_GRID_HEIGHT - VISIBLE_GRID_HEIGHT, self.y))

    def show_full_arena(self):
        """Position camera to show full arena"""
        self.x = 0
        self.y = 0
        self.scale = min(
            VISIBLE_GRID_WIDTH / WORLD_GRID_WIDTH, VISIBLE_GRID_HEIGHT / WORLD_GRID_HEIGHT
        )

    def reset_zoom(self):
        """Reset to normal zoom level"""
        self.scale = 1

    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates"""
        if self.scale == 1:
            # Normal view
            return ((pos[0] - self.x) * CELL_SIZE, (pos[1] - self.y) * CELL_SIZE)
        else:
            # Zoomed out view
            scaled_cell_size = CELL_SIZE * self.scale
            return (pos[0] * scaled_cell_size, pos[1] * scaled_cell_size)

    def is_visible(self, pos):
        """Check if a position is within the visible area"""
        if self.scale == 1:
            # Normal view
            return (
                self.x <= pos[0] < self.x + VISIBLE_GRID_WIDTH
                and self.y <= pos[1] < self.y + VISIBLE_GRID_HEIGHT
            )
        else:
            # Zoomed out - all positions are visible
            return True


def create_gradient_background(width, height, top_color, bottom_color):
    """Create a vertical gradient background."""
    bg_surface = pygame.Surface((width, height))
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        pygame.draw.line(bg_surface, (r, g, b), (0, y), (width, y))
    return bg_surface


def draw_snake_segment(
    screen, pos, cell_color, border_color, is_head=False, direction=RIGHT, camera=None
):
    """Draw a snake segment with camera transformation"""
    if camera and not camera.is_visible(pos) and camera.scale == 1:
        return  # Skip drawing if not visible and not zoomed out

    screen_pos = camera.world_to_screen(pos) if camera else (pos[0] * CELL_SIZE, pos[1] * CELL_SIZE)
    x, y = screen_pos

    # Calculate size based on camera scale
    size = CELL_SIZE * camera.scale if camera else CELL_SIZE
    size = max(1, size)  # Ensure minimum size

    cell_rect = pygame.Rect(x, y, size, size)

    # For very small cells, just draw a colored pixel
    if size <= 3:
        pygame.draw.rect(screen, cell_color, cell_rect)
        return

    # Draw shadow
    shadow_rect = cell_rect.copy()
    shadow_rect.move_ip(min(3, size / 6), min(3, size / 6))
    pygame.draw.rect(screen, (0, 0, 0), shadow_rect, border_radius=max(1, int(size / 3)))

    # Draw segment
    pygame.draw.rect(screen, cell_color, cell_rect, border_radius=max(1, int(size / 3)))

    if size >= 10:  # Only draw border and eyes if big enough
        pygame.draw.rect(screen, border_color, cell_rect, 2, border_radius=max(1, int(size / 3)))

        if is_head and size >= 12:
            # Draw eyes based on direction
            eye_radius = max(1, int(size / 6))
            cx = x + size // 2
            cy = y + size // 2

            if direction == UP:
                eye_offset_y = -size // 4
                offset = size // 6
                eye1 = (cx - offset, cy + eye_offset_y)
                eye2 = (cx + offset, cy + eye_offset_y)
            elif direction == DOWN:
                eye_offset_y = size // 4
                offset = size // 6
                eye1 = (cx - offset, cy + eye_offset_y)
                eye2 = (cx + offset, cy + eye_offset_y)
            elif direction == LEFT:
                eye_offset_x = -size // 4
                offset = size // 6
                eye1 = (cx + eye_offset_x, cy - offset)
                eye2 = (cx + eye_offset_x, cy + offset)
            else:  # RIGHT
                eye_offset_x = size // 4
                offset = size // 6
                eye1 = (cx + eye_offset_x, cy - offset)
                eye2 = (cx + eye_offset_x, cy + offset)

            pygame.draw.circle(screen, (255, 255, 255), eye1, eye_radius)
            pygame.draw.circle(screen, (0, 0, 0), eye1, max(1, eye_radius // 2))
            pygame.draw.circle(screen, (255, 255, 255), eye2, eye_radius)
            pygame.draw.circle(screen, (0, 0, 0), eye2, max(1, eye_radius // 2))


def draw_food(screen, pos, camera=None):
    """Draw food with camera transformation"""
    if camera and not camera.is_visible(pos) and camera.scale == 1:
        return  # Skip drawing if not visible and not zoomed out

    screen_pos = camera.world_to_screen(pos) if camera else (pos[0] * CELL_SIZE, pos[1] * CELL_SIZE)
    x, y = screen_pos

    # Calculate size based on camera scale
    size = CELL_SIZE * camera.scale if camera else CELL_SIZE
    size = max(1, size)  # Ensure minimum size

    if size <= 3:
        # For very small cells, just draw a colored point
        pygame.draw.rect(screen, FOOD_COLOR, (x, y, size, size))
        return

    food_rect = pygame.Rect(x, y, size, size)
    shadow_rect = food_rect.copy()
    shadow_rect.move_ip(min(3, size / 6), min(3, size / 6))

    pygame.draw.ellipse(screen, (0, 0, 0), shadow_rect)
    pygame.draw.ellipse(screen, FOOD_COLOR, food_rect)

    if size >= 10:  # Only draw border if big enough
        pygame.draw.ellipse(screen, BORDER_COLOR, food_rect, 2)


def get_random_food_position(snakes):
    """Return a random grid position not occupied by any snake"""
    while True:
        x = random.randint(0, WORLD_GRID_WIDTH - 1)
        y = random.randint(0, WORLD_GRID_HEIGHT - 1)
        pos = (x, y)

        occupied = False
        for snake in snakes:
            if pos in snake.segments:
                occupied = True
                break

        if not occupied:
            return pos


def draw_stats(screen, snakes, food_count):
    """Draw statistics overlay"""
    font = pygame.font.SysFont("Arial", 16)

    # Calculate statistics
    alive_count = sum(1 for snake in snakes if snake.alive)
    max_gen = max(snake.generation for snake in snakes) if snakes else 0

    # Draw statistics text
    stats = [
        f"Generation: {max_gen}",
        f"Snakes Alive: {alive_count}/{len(snakes)}",
        f"Food Available: {food_count}",
    ]

    y = 10
    for stat in stats:
        text = font.render(stat, True, (255, 255, 255))
        screen.blit(text, (10, y))
        y += 20


def draw_snake_genetics(screen, snake):
    """Draw genetic information for the currently followed snake"""
    if not snake:
        return

    font = pygame.font.SysFont("Arial", 14)
    genetics_bg = pygame.Surface((260, 150))
    genetics_bg.set_alpha(180)
    genetics_bg.fill((0, 0, 0))
    screen.blit(genetics_bg, (WIDTH - 270, 10))

    title = font.render(f"Snake #{snake.id} (Gen {snake.generation})", True, (255, 255, 255))
    screen.blit(title, (WIDTH - 260, 15))

    y = 40
    for gene, value in snake.ai.genes.items():
        text = font.render(f"{gene}: {value:.2f}", True, (220, 220, 220))
        screen.blit(text, (WIDTH - 260, y))
        y += 18

    # Stats
    stats_text = [
        f"Length: {len(snake.segments)}",
        f"Food eaten: {snake.ai.food_eaten}",
        f"Age: {snake.ai.age}",
        f"Kills: {snake.ai.kills}",
    ]

    y += 5
    for stat in stats_text:
        text = font.render(stat, True, (180, 255, 180))
        screen.blit(text, (WIDTH - 260, y))
        y += 18


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Evolutionary Snake Battle")

    bg_surface = create_gradient_background(WIDTH, HEIGHT, BG_TOP_COLOR, BG_BOTTOM_COLOR)

    # Initialize camera
    camera = Camera()

    # Initialize gene tracker
    gene_tracker = GeneTracker()

    # Initialize starting snakes
    snakes = [
        Snake(
            [(5, WORLD_GRID_HEIGHT // 2), (4, WORLD_GRID_HEIGHT // 2), (3, WORLD_GRID_HEIGHT // 2)],
            RIGHT,
        ),
        Snake(
            [
                (WORLD_GRID_WIDTH - 6, WORLD_GRID_HEIGHT // 2),
                (WORLD_GRID_WIDTH - 5, WORLD_GRID_HEIGHT // 2),
                (WORLD_GRID_WIDTH - 4, WORLD_GRID_HEIGHT // 2),
            ],
            LEFT,
        ),
    ]

    # Initialize food (multiple food items)
    food_count = max(10, len(snakes) // 2)
    food = [get_random_food_position(snakes) for _ in range(food_count)]

    # Game loop
    running = True
    paused = False
    simulation_speed = 10  # Frames per movement
    frame_counter = 0
    follow_snake_index = 0
    zoomed_out = False
    show_graphs = False
    graphs_surface = None

    while running:
        food_count = max(10, len(snakes) // 2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    simulation_speed = max(1, simulation_speed - 1)
                elif event.key == pygame.K_DOWN:
                    simulation_speed = min(60, simulation_speed + 1)
                elif event.key == pygame.K_TAB:
                    # Switch which snake to follow and ensure we're not zoomed out
                    zoomed_out = False
                    camera.reset_zoom()
                    show_graphs = False  # Close graphs if open
                    alive_snakes = [s for s in snakes if s.alive]
                    if alive_snakes:
                        follow_snake_index = (follow_snake_index + 1) % len(alive_snakes)
                elif event.key == pygame.K_ESCAPE:
                    # Toggle zoomed out view
                    zoomed_out = not zoomed_out
                    show_graphs = False  # Close graphs if open
                    if zoomed_out:
                        camera.show_full_arena()
                    else:
                        camera.reset_zoom()
                elif event.key == pygame.K_g:
                    # Toggle genetic graphs
                    show_graphs = not show_graphs
                    if show_graphs:
                        # Create the graphs
                        graphs_surface = gene_tracker.create_all_graphs()
                        # Pause the game when viewing graphs
                        paused = True
                    else:
                        # Resume the game when closing graphs
                        paused = False

            # Mouse wheel for scrolling (only when not zoomed out)
            elif event.type == pygame.MOUSEBUTTONDOWN and not zoomed_out and not show_graphs:
                if event.button == 4:  # Scroll up
                    camera.y = max(0, camera.y - 3)
                elif event.button == 5:  # Scroll down
                    camera.y = min(WORLD_GRID_HEIGHT - VISIBLE_GRID_HEIGHT, camera.y + 3)

        # Update logic at reduced rate
        if not paused:
            frame_counter += 1

            if frame_counter >= simulation_speed:
                frame_counter = 0

                # Move each snake
                eaten_food = []
                for snake in snakes:
                    if snake.alive:
                        snake.ai.age += 1
                        food_pos = snake.move(food, [s for s in snakes if s != snake])
                        if food_pos:
                            eaten_food.append(food_pos)

                # Remove eaten food
                for pos in eaten_food:
                    if pos in food:
                        food.remove(pos)

                # Add new food to maintain food count
                while len(food) < food_count:
                    food.append(get_random_food_position(snakes))

                def convert_dead_snake_to_food(snake, food_list):
                    """Convert every other segment of a dead snake into food"""
                    # Skip the head (index 0) and take every other segment (1, 3, 5, etc.)
                    for i in range(1, len(snake.segments), 2):
                        food_list.append(snake.segments[i])
                    return food_list

                # Check collisions
                for snake in snakes:
                    if snake.alive:
                        snake.check_collision([s for s in snakes if s != snake])
                        # If snake just died, convert its body to food
                        if not snake.alive:
                            food = convert_dead_snake_to_food(snake, food)

                # Check reproduction
                new_snakes = []
                for snake in snakes:
                    if snake.alive and snake.should_reproduce():
                        offspring = snake.reproduce()
                        new_snakes.extend(offspring)
                        snake.alive = False  # Parent dies after reproduction

                # Add new snakes to the list
                snakes.extend(new_snakes)

                # Remove dead snakes (but keep a minimum number)
                alive_snakes = [s for s in snakes if s.alive]
                if len(alive_snakes) < 2 and len(snakes) > 0:
                    # Create new snakes if population is too low
                    while len(alive_snakes) < 2:
                        # Create new snake at random position
                        x = random.randint(0, WORLD_GRID_WIDTH - 1)
                        y = random.randint(0, WORLD_GRID_HEIGHT - 1)

                        # Create new snake
                        new_snake = Snake([(x, y), (x - 1, y), (x - 2, y)], RIGHT)
                        snakes.append(new_snake)
                        alive_snakes.append(new_snake)
                else:
                    # Clean up dead snakes
                    snakes = [s for s in snakes if s.alive]

                # Update genetic tracking stats
                gene_tracker.update_stats(snakes)

                # Update graphs if they're being shown
                if show_graphs:
                    graphs_surface = gene_tracker.create_all_graphs()

        # If we're showing graphs, just display them and skip other rendering
        if show_graphs and graphs_surface:
            screen.fill((30, 30, 40))
            screen.blit(graphs_surface, (0, 0))
            pygame.display.flip()
            clock.tick(60)
            continue

        # Center camera on a snake if available and not zoomed out
        alive_snakes = [s for s in snakes if s.alive]
        followed_snake = None

        if alive_snakes and not zoomed_out:
            # If we were following a snake, try to keep following it
            if follow_snake_index < len(snakes) and snakes[follow_snake_index].alive:
                followed_snake = snakes[follow_snake_index]
            else:
                # Our snake died, pick the first available live snake
                follow_snake_index = snakes.index(alive_snakes[0])
                followed_snake = alive_snakes[0]
            camera.center_on(followed_snake.segments[0])

        # Draw everything
        screen.blit(bg_surface, (0, 0))

        # Draw world grid (only in zoomed out mode)
        if zoomed_out:
            # Draw a border around the entire world
            world_rect = pygame.Rect(
                0,
                0,
                WORLD_GRID_WIDTH * CELL_SIZE * camera.scale,
                WORLD_GRID_HEIGHT * CELL_SIZE * camera.scale,
            )
            pygame.draw.rect(screen, (100, 100, 100), world_rect, 1)
        else:
            # Draw grid lines
            for x in range(0, WIDTH, CELL_SIZE):
                pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, HEIGHT))
            for y in range(0, HEIGHT, CELL_SIZE):
                pygame.draw.line(screen, (50, 50, 50), (0, y), (WIDTH, y))

        # Draw food
        for f in food:
            draw_food(screen, f, camera)

        # Draw snakes
        for snake in snakes:
            if snake.alive:
                # Draw head
                draw_snake_segment(
                    screen,
                    snake.segments[0],
                    snake.color_head,
                    BORDER_COLOR,
                    is_head=True,
                    direction=snake.direction,
                    camera=camera,
                )

                # Draw body
                for pos in snake.segments[1:]:
                    draw_snake_segment(
                        screen, pos, snake.color_body, BORDER_COLOR, is_head=False, camera=camera
                    )

        # Draw statistics
        draw_stats(screen, snakes, len(food))

        # Draw genetics of followed snake if not zoomed out
        if followed_snake and not zoomed_out:
            draw_snake_genetics(screen, followed_snake)

        # Draw UI info
        font = pygame.font.SysFont("Arial", 16)
        speed_text = font.render(
            f"Speed: {simulation_speed} (↑/↓ to adjust)", True, (200, 200, 200)
        )
        pause_text = font.render(
            "PAUSED (Space to resume)" if paused else "Space to pause", True, (200, 200, 200)
        )
        graphs_text = font.render("G to view genetic evolution graphs", True, (200, 200, 200))

        if zoomed_out:
            follow_text = font.render(
                "Full Arena View (ESC to toggle, TAB to follow a snake)", True, (200, 200, 200)
            )
        else:
            follow_text = font.render(
                f"Following Snake #{followed_snake.id if followed_snake else '?'} (TAB to switch, ESC for full view)",
                True,
                (200, 200, 200),
            )

        screen.blit(speed_text, (10, HEIGHT - 80))
        screen.blit(pause_text, (10, HEIGHT - 60))
        screen.blit(follow_text, (10, HEIGHT - 40))
        screen.blit(graphs_text, (10, HEIGHT - 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
