# Author: Marlos C. Machado

from agent import Agent
import numpy as np


class QLearning_Joint(Agent):

    def __init__(self, env, step_size, gamma, epsilon, q_agent, q_count):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = step_size
        self.epsilon = epsilon
        self.curr_s = self.env.get_current_state()
        self.q = q_agent.get_q() + q_count.get_q()
        self.state_visitation_count = np.ones((self.env.get_size_states(), self.env.get_size_actions()))

    def step(self):
        curr_a = self.epsilon_greedy(self.q[self.curr_s], epsilon=self.epsilon)
        r = self.env.act(curr_a)
        next_s = self.env.get_current_state()
        next_a = self.epsilon_greedy(self.q[next_s], epsilon=self.epsilon)

        self.curr_s = next_s
        self.current_undisc_return += r

        if self.env.is_terminal():
            self.episode_count += 1
            self.total_undisc_return += self.current_undisc_return
            self.current_undisc_return = 0

    def update_q_values(self, s, a, r, next_s, next_a):
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * (1.0 - self.env.is_terminal()) *
                                                    max(self.q[next_s]) - self.q[s][a])
