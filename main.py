import pygame
from maze import Maze
from agent import Agent
from environment import Environment
from train import random_action
from train import play_random_episode
import torch
import os
from dqn import DQN
from dqn_agent import DQNAgent, ACTIONS
import matplotlib.pyplot as plt

env = Environment()
dqn_agent = DQNAgent()

MODEL_PATH = "maze_dqn.pth"

if os.path.exists(MODEL_PATH):

    dqn_agent.model.load_state_dict(
        torch.load(MODEL_PATH)
    )

    dqn_agent.target_model.load_state_dict(
        dqn_agent.model.state_dict()
    )

    print("Loaded trained model.")

else:

    print("No saved model found.")



pygame.init()


WIDTH = env.maze.cols * env.maze.cell_size
HEIGHT = env.maze.rows * env.maze.cell_size

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Maze Reinforcement Learning")

clock = pygame.time.Clock()

running = True

state = torch.tensor([1.0, 1.0])

with torch.no_grad():
    q_values = dqn_agent.model(state)

print("Q-values:", q_values)

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            # -------------------------
            # Human Controls
            # -------------------------
            if event.key == pygame.K_UP:
                state, reward, done = env.step("UP")

            elif event.key == pygame.K_DOWN:
                state, reward, done = env.step("DOWN")

            elif event.key == pygame.K_LEFT:
                state, reward, done = env.step("LEFT")

            elif event.key == pygame.K_RIGHT:
                state, reward, done = env.step("RIGHT")

            # -------------------------
            # Random Agent (Press R)
            # -------------------------
            elif event.key == pygame.K_r:

                action = random_action()

                state, reward, done = env.step(action)

                print("\n Random Agent")
                print("Action :", action)
            
            elif event.key == pygame.K_d:

                print("\n===== DQN TEST =====\n")

                state = env.reset()

                done = False

                steps = 0

                MAX_STEPS = 100

                while not done and steps < MAX_STEPS:

                    action_index = dqn_agent.choose_best_action(state)

                    action = ACTIONS[action_index]

                    state, reward, done = env.step(action)

                    screen.fill((255,255,255))

                    env.maze.draw(screen)

                    env.agent.draw(
                        screen,
                        env.maze.cell_size
                    )

                    pygame.display.flip()

                    pygame.time.delay(200)

                    print(
                        f"Step {steps:2d}"
                        f" | Action: {action:5s}"
                        f" | Reward: {reward:4d}"
                        f" | State: {state}"
                    )

                    steps += 1

                print("\nFinished.")

                print("Steps :", steps)

                print("Done  :", done)

            
            elif event.key == pygame.K_t:

                NUM_EPISODES = 1000

                print("\n========== TRAINING ==========\n")

                reward_history = []

                for episode in range(NUM_EPISODES):

                    state = env.reset()

                    done = False

                    total_reward = 0

                    while not done:

                        action = dqn_agent.choose_action(state)

                        action_string = ACTIONS[action]

                        next_state, reward, done = env.step(action_string)

                        dqn_agent.remember(
                            state,
                            action,
                            reward,
                            next_state,
                            done
                        )

                        dqn_agent.train_step()

                        state = next_state

                        total_reward += reward

                    if dqn_agent.epsilon > dqn_agent.epsilon_min:

                        reward_history.append(total_reward)

                        dqn_agent.epsilon *= dqn_agent.epsilon_decay

                    if (episode + 1) % 100 == 0:

                        print(
                                f"Episode {episode+1:4d}/{NUM_EPISODES}"
                                f" | Reward: {total_reward:5d}"
                                f" | Memory: {dqn_agent.memory_size():5d}"
                                f" | Epsilon: {dqn_agent.epsilon:.3f}"
                            
                        )

                torch.save(
                    dqn_agent.model.state_dict(),
                    MODEL_PATH
                )

                print("Model saved.")
                print("\nTraining Finished!")

                plt.figure(figsize=(8,5))

                plt.plot(reward_history)

                plt.xlabel("Episode")

                plt.ylabel("Reward")

                plt.title("Training Reward")

                plt.grid(True)

                plt.show()

            elif event.key == pygame.K_a:
                play_random_episode(env, screen)
                continue

            else:
                continue

            print("State :", state)
            print("Reward:", reward)
            print("Done  :", done)
            print()

    screen.fill((255,255,255))

    env.maze.draw(screen)
    env.agent.draw(screen, env.maze.cell_size)
    

    pygame.display.flip()

    clock.tick(60)

pygame.quit()