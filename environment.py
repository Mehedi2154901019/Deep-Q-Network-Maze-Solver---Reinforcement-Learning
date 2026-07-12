from maze import Maze
from agent import Agent


class Environment:

    def __init__(self):

        self.maze = Maze()
        self.agent = Agent()

    def reset(self):

        self.agent.row = 1
        self.agent.col = 1

        self.agent.path = [(1, 1)]

        return self.get_state()

    def get_state(self):

        return (
            self.agent.row / (self.maze.rows - 1),
            self.agent.col / (self.maze.cols - 1)
        )

    def step(self, action):

        reward = self.agent.move(action, self.maze)

        state = self.get_state()

        done = (
            self.maze.layout[
                self.agent.row
            ][
                self.agent.col
            ] == "G"
        )

        return state, reward, done