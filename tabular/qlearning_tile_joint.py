from agent import Agent
import numpy as np
from tiles import IHT, generate_tiles


class QLearning_Tile_Joint(Agent):

    def __init__(self, env, step_size, gamma, epsilon, beta=1000.0, agg_func=np.max):
        super().__init__(env)
        self.gamma = gamma
        self.beta = beta
        self.alpha = step_size
        self.epsilon = epsilon
        self.curr_s = self.env.get_current_state()

        self.maxSize = 3
        self.iht = IHT(self.maxSize)
        self.numTilings = 4
        self.tile_visitation_count = np.ones((self.numTilings, self.maxSize, self.env.get_size_actions()))
        self.state_tiles = np.zeros((self.env.get_size_states(), self.numTilings), dtype=int)

        self.agg_func = agg_func
        self.state_tiles = [[0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 1, 0],
                            [1, 1, 1, 0],
                            [1, 2, 1, 0],
                            [1, 2, 2, 0]]
        self.create_tilings()

        self.q_count = np.zeros((self.env.get_size_states(), self.env.get_size_actions())) 

    def print_tiling_counts(self):
        print(self.tile_visitation_count)

    def create_tilings(self):
        for state in self.env.state_set:
            tiles = generate_tiles(self.iht, self.numTilings, [state])
            self.state_tiles[state] = tiles

    def add_tile_counts(self, s, a):
        for tiling in range(self.numTilings):
            tile = self.state_tiles[s][tiling]
            self.tile_visitation_count[tiling][tile][a] += 1

    def step(self):
        curr_a = self.epsilon_greedy(self.q[self.curr_s] + self.q_count[self.curr_s], epsilon=self.epsilon)

        #Calculate environmental reward and Intrinsic Reward
        r = self.env.act(curr_a)
        r_intrinsic = self.get_reward_count(self.curr_s, curr_a)


        self.add_tile_counts(self.curr_s, curr_a)

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

    def get_reward_count(self, curr_s, curr_a):
        n_s = np.zeros(self.env.get_size_actions())
        counts = np.zeros((self.numTilings, self.env.get_size_actions()))

        for tiling in range(self.numTilings):
            tile = self.state_tiles[curr_s][tiling]
            counts[tiling] = self.tile_visitation_count[tiling][tile]

        n_s = self.agg_func(counts, axis=0)
        return self.beta * np.sqrt((2*np.log(np.sum(n_s)))/(n_s[curr_a]))


    def update_q_values(self, s, a, r, next_s):
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * (1.0 - self.env.is_terminal()) *
                                                    max(self.q[next_s]) - self.q[s][a]) 
        
    def update_q_counts(self, s, a, r, next_s):
         self.q_count[s][a] = self.q_count[s][a] + self.alpha * (r + self.gamma * (1.0 - self.env.is_terminal()) *
                                                    max(self.q_count[next_s]) - self.q_count[s][a]) 
        
    def get_visitation_count(self):
        return self.state_action_visitation_count
