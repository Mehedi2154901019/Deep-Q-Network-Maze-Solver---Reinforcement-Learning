import pygame


class Maze:

    def __init__(self):

        self.cell_size = 40

        self.layout = [

            "####################",
            "#S.................#",
            "#.######.#########.#",
            "#........#.........#",
            "#.########.######..#",
            "#................G.#",
            "####################"

        ]

        self.rows = len(self.layout)
        self.cols = len(self.layout[0])

    def draw(self, screen):

        colors = {
            "#": (40, 40, 40),       # Wall
            ".": (240, 240, 240),   # Floor
            "S": (100, 220, 100),   # Start
            "G": (220, 70, 70)      # Goal
        }

        for row in range(self.rows):

            for col in range(self.cols):

                cell = self.layout[row][col]

                color = colors[cell]

                pygame.draw.rect(

                    screen,

                    color,

                    (

                        col * self.cell_size,
                        row * self.cell_size,
                        self.cell_size,
                        self.cell_size

                    )

                )

                pygame.draw.rect(

                    screen,

                    (180, 180, 180),

                    (

                        col * self.cell_size,
                        row * self.cell_size,
                        self.cell_size,
                        self.cell_size

                    ),

                    1

                )