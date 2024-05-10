# Author: Marlos C. Machado

from agent import Agent
import numpy as np


class QLearning_Joint(Agent):

    def __init__(self, env, step_size, gamma, epsilon, beta=100.0):
        super().__init__(env)
        self.gamma = gamma
        self.beta = beta
        self.alpha = step_size
        self.epsilon = epsilon
        self.curr_s = self.env.get_current_state()
        self.state_action_visitation_count = np.ones((self.env.get_size_states(), self.env.get_size_actions()))
        self.q_count = np.zeros((self.env.get_size_states(), self.env.get_size_actions())) 

        # self.state_visitation_count = np.array([0, 0, 0, 0, 0, 0])

    def step(self):
        curr_a = self.epsilon_greedy(self.q[self.curr_s] + self.q_count[self.curr_s], epsilon=self.epsilon)

        #Calculate environmental reward and Intrinsic Reward
        r = self.env.act(curr_a)
        r_intrinsic = self.get_reward_count(self.curr_s, curr_a)

        # print("state:", self.curr_s, "action:", curr_a)
        # print("reward:",r, "Intrinsic reward:", r_intrinsic)


        self.state_action_visitation_count[self.curr_s][curr_a] += 1

        next_s = self.env.get_current_state()
        # next_a = self.epsilon_greedy(self.q[next_s], epsilon=self.epsilon)
        

        self.update_q_values(self.curr_s, curr_a, r          , next_s)
        self.update_q_counts(self.curr_s, curr_a, r_intrinsic, next_s)

        # print(self.q)
        # print(self.q_count)

        self.curr_s = next_s

        self.current_undisc_return += r
        if self.env.is_terminal():
            self.episode_count += 1
            self.total_undisc_return += self.current_undisc_return
            self.current_undisc_return = 0

    def get_reward_count(self,s, a):
            return self.beta * np.sqrt(2*np.log(np.sum(self.state_action_visitation_count[s]))/(self.state_action_visitation_count[s][a]))

    def get_q_count(self, s):
        count = self.state_action_visitation_count[s]
        return self.beta / np.sqrt(count + 1)

    def update_q_values(self, s, a, r, next_s):
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * (1.0 - self.env.is_terminal()) *
                                                    max(self.q[next_s]) - self.q[s][a]) 
        
    def update_q_counts(self, s, a, r, next_s):
         self.q_count[s][a] = self.q_count[s][a] + self.alpha * (r + self.gamma * (1.0 - self.env.is_terminal()) *
                                                    max(self.q_count[next_s]) - self.q_count[s][a]) 
        
    def get_visitation_count(self):
        return self.state_action_visitation_count
