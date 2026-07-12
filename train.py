import random
import time

import pygame

ACTIONS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT"
]


def random_action():
    return random.choice(ACTIONS)


def run_random_episode(env, max_steps=100):

    state = env.reset()

    done = False

    total_reward = 0

    step = 0

    while not done and step < max_steps:

        action = random_action()

        next_state, reward, done = env.step(action)

        print(
            f"Step: {step:3d} | "
            f"State: {state} | "
            f"Action: {action:5s} | "
            f"Reward: {reward:4d} | "
            f"Next: {next_state} | "
            f"Done: {done}"
        )

        state = next_state

        total_reward += reward

        step += 1

    print("\nEpisode Finished")
    print("Total Reward:", total_reward)
    print("Steps:", step)


def play_random_episode(env, screen):

    state = env.reset()

    done = False

    total_reward = 0

    step = 0

    MAX_STEPS = 300

    while not done and step < MAX_STEPS:

        # Handle quit events so the window doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        action = random_action()

        state, reward, done = env.step(action)

        total_reward += reward

        print(
            f"Step {step:3d} | "
            f"Action: {action:5s} | "
            f"Reward: {reward:4d} | "
            f"State: {state}"
        )

        # Draw updated environment
        screen.fill((255,255,255))
        env.maze.draw(screen)
        env.agent.draw(screen, env.maze.cell_size)

        pygame.display.flip()

        time.sleep(0.15)

        step += 1

    print("\nEpisode Finished")
    print("Total Reward:", total_reward)