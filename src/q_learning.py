import numpy as np
from pyro import do
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
                min_epsilon=0.1,
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




    def get_action(self, state, rank = 0):
        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(len(self.actions))
            action = list(self.actions.values())[action_index]  
        else:
            x, y = state
            # Sort Q-values in descending order and get the (rank)th best action
            sorted_indices = np.argsort(self.q_values[x, y, :])[::-1]
            action_index = sorted_indices[min(rank, len(sorted_indices) - 1)]
            action = list(self.actions.values())[action_index]
        return action, action_index
    
    def distance_to_goal(self, state):
        x, y = state
        return np.sqrt((x - self.top[0]) ** 2 + (y - self.top[1]) ** 2)


    def train(self, start_state, episodes=1000, max_steps=100, 
              reward_f= lambda s, s_prime, env: env.get_reward(s_prime), 
              risk_eval=lambda s, a, env: (0, True),
              reward_risk_p_op=lambda reward, risk_p: reward + risk_p
              ):

        self.q_values = np.zeros((self.env.grid_shape[0], self.env.grid_shape[1], len(self.actions)))  # reset Q-values at the start of training
        # continuous training the Q_learning agent.
        # resamples visited states per run, returns the training data for the GP (state, action) -> next state
        # if the agent takes an unsafe action, it will be punished for the risk, r=0 and combined with risk punisment,
        #  but will not transition to the next state (as if the shield prevented the action)

        # N samples x M tasks, (0,4) as we have not sampled yet
        train_x = torch.empty((0, 4), dtype=torch.float64)
        train_y = torch.empty((0, 2), dtype=torch.float64)

        unsafe_counts = 0

        for episode in tqdm(range(episodes), desc="Training Episodes"):
            s = start_state
            rank = 0

            for step in range(max_steps):
                x, y = s

                a, a_index = self.get_action(s, rank=rank)
                ps = []

                r_p, safe = risk_eval(s, a, self.env)

                while not safe:
                    ps.append((a, a_index, r_p))
                    rank += 1
                    a, a_index = self.get_action(s, rank=rank)
                    r_p, safe = risk_eval(s, a, self.env)
                    if rank >= len(self.actions):
                        a, a_index, r_p = sorted(ps, key=lambda x: x[2])[-1]   
                        safe = True
                        
                    r = reward_risk_p_op(0, r_p)


                    # if action not taken, still punish the agent for the risk, but do not transition to the next state
                    # for Q(a', s') use a punishment as we assume the next state is unsafe and needs punishment as well
                    self.q_values[x, y, a_index] = (
                        (1 - self.alpha) * self.q_values[x, y, a_index]
                        + self.alpha * (r + self.gamma * r_p)
                    )          



                s_prime = self.env.transition(s, a)

                r = reward_risk_p_op(reward_f(s, s_prime, self.env), r_p)

                # states_visited[s_prime] = states_visited.get(s_prime, 0) + 1
                x_prime, y_prime = s_prime
                if self.env.unsafe_mask[s_prime[0], s_prime[1]]:
                    unsafe_counts += 1


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

        return train_x, train_y, unsafe_counts

    def get_q_values(self):
        return self.q_values
    





