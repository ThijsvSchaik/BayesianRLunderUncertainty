import numpy as np
import torch
from tqdm import tqdm

class QLearningAgent:
    def __init__(self, 
                 env, 
                 actions = {
                        'up': (0, -1),
                        'right': (1, 0),
                        'down': (0, 1),
                        'left': (-1, 0),
                    }, 
                alpha=0.1, 
                gamma=0.9, 
                epsilon=0.1,
                epsilon_decay=0.995,
                min_epsilon=0.1
                ):
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = np.zeros((env.grid_shape[0], env.grid_shape[1], len(actions)))
        self.actions = actions
        self.top = env.reward_gaussian_mean
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay




    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(len(self.actions))
            action = list(self.actions.values())[action_index]  
        else:
            x, y = state
            action_index = np.argmax(self.q_values[x, y, :])
            action = list(self.actions.values())[action_index]
        return action, action_index

    def square_keep_sign(self, n):
        if n >= 0:
            return n ** 2
        else:
            return -(n ** 2)

    def distance_to_goal(self, state):
        x, y = state
        return np.sqrt((x - self.top[0]) ** 2 + (y - self.top[1]) ** 2)
        
    def train(self, start_state, episodes=1000, max_steps=100):
        # continuous training the Q_learning agent.
        # resamples visited states per run, returns the training data for the GP (state, action) -> next state

        # N samples x M tasks, (0,4) as we have not sampled yet
        train_x = torch.empty((0, 4), dtype=torch.float64)
        train_y = torch.empty((0, 2), dtype=torch.float64)

        # states_visited = {}

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            s = start_state

            for step in range(max_steps):
                x, y = s

                a, a_index = self.get_action(s)
                s_prime = self.env.transition(s, a)

                if self.distance_to_goal(s_prime) < 5:
                    r = 100
                else:
                    # r = env.get_reward(s_prime) - 10
                    # r = distance_to_goal(s_prime) * -2 + r
                    r = self.square_keep_sign(self.distance_to_goal(s) - self.distance_to_goal(s_prime)) - 10

                # states_visited[s_prime] = states_visited.get(s_prime, 0) + 1

                x_prime, y_prime = s_prime


                best_next_q = np.max(self.q_values[x_prime, y_prime, :])

                # Bellman equation
                # $Q(s, a) = (1 - \alpha) Q(s, a) + \alpha (r + \gamma \max_{a'} Q(s', a'))$
                self.q_values[x, y, a_index] = (
                    (1 - self.alpha) * self.q_values[x, y, a_index]
                    + self.alpha * (r + self.gamma * best_next_q)
                )

                # if step == MAX_STEPS - 1 and episode % 100 == 0:
                #     print(f"Episode {episode+1}/{EPISODES} ended at max steps. Last state: {s_prime}, reward: {r:.2f}")

                # get training data for the GP
                train_x = torch.cat((train_x, torch.tensor([[x, y, a[0], a[1]]], dtype=torch.float64)), dim=0)
                train_y = torch.cat((train_y, torch.tensor([[x_prime, y_prime]], dtype=torch.float64)), dim=0)

                s = s_prime

                if self.distance_to_goal(s) < 5:
                    break



            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return train_x, train_y

    def get_q_values(self):
        return self.q_values
    





