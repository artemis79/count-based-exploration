from agent import Agent
import numpy as np


class QLearning_Count(Agent):

    def __init__(self, env, step_size, gamma, epsilon, beta=100.0):
        super().__init__(env)
        self.gamma = gamma
        self.beta = beta
        self.alpha = step_size
        self.epsilon = epsilon
        self.curr_s = self.env.get_current_state()
        self.state_action_visitation_count = np.ones((self.env.get_size_states(), self.env.get_size_actions()))

        # self.state_visitation_count = np.array([0, 0, 0, 0, 0, 0])

    def step(self):
        curr_a = self.epsilon_greedy(self.q[self.curr_s], epsilon=self.epsilon)
        r = self.env.act(curr_a)
        r_intrinsic = self.get_reward_count(self.curr_s, curr_a)
        self.state_action_visitation_count[self.curr_s][curr_a] += 1

        next_s = self.env.get_current_state()
        next_a = self.epsilon_greedy(self.q[next_s], epsilon=self.epsilon)
        

        self.update_q_values(self.curr_s, curr_a, r+r_intrinsic, next_s, next_a)

        self.curr_s = next_s

        self.current_undisc_return += r
        if self.env.is_terminal():
            self.episode_count += 1
            self.total_undisc_return += self.current_undisc_return
            self.current_undisc_return = 0

    def get_reward_count(self, s, a):
        n_s = self.state_action_visitation_count[s]
        r = self.beta * np.sqrt(2*np.log(np.sum(n_s))/n_s[a])
        return r

    def update_q_values(self, s, a, r, next_s, next_a):
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * (1.0 - self.env.is_terminal()) *
                                                    max(self.q[next_s]) - self.q[s][a]) 
        
    def get_visitation_count(self):
        return self.state_action_visitation_count
