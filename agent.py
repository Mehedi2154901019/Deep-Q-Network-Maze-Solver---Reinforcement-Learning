import pygame


class Agent:

    def __init__(self):

        self.row = 1
        self.col = 1

        self.color = (50, 120, 255)
        self.path = []

    def draw(self, screen, cell_size):

        padding = 6

        for row, col in self.path:

            pygame.draw.circle(

                screen,

                (255, 200, 0),

                (

                    col * cell_size + cell_size // 2,

                    row * cell_size + cell_size // 2

                ),

                4

            )
        pygame.draw.rect(

            screen,

            self.color,

            (

                self.col * cell_size + padding,

                self.row * cell_size + padding,

                cell_size - padding * 2,

                cell_size - padding * 2

            )

        )

    def move(self, direction, maze):

        new_row = self.row
        new_col = self.col

        self.path.append(
            (self.row, self.col)
        )

        if direction == "UP":
            new_row -= 1

        elif direction == "DOWN":
            new_row += 1

        elif direction == "LEFT":
            new_col -= 1

        elif direction == "RIGHT":
            new_col += 1

        # Hit a wall
        if maze.layout[new_row][new_col] == "#":
            return -5

        # Valid move
        self.row = new_row
        self.col = new_col

        # Goal reached
        if maze.layout[self.row][self.col] == "G":
            return 100

        # Normal move
        return -1