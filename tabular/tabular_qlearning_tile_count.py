import utils
import random
import numpy as np
import environment as env
from qlearning_tile_count import QLearning_Tile_Count
from tqdm import tqdm

actions = ["right", "left"]

if __name__ == "__main__":
    # Read arguments:
    number_of_run = 100
    args = utils.ArgsParser.read_input_args()

    environment = env.MDP(args.input)

    undisc_return = np.zeros((100000-2, number_of_run))
    bonus_rewards = np.zeros((6, 2, 100000-2, number_of_run))

    for run in tqdm(range(0, number_of_run)):
        random.seed(run)
        np.random.seed(run)
        for ep in range(args.num_episodes):
            qlearning_tile_count= QLearning_Tile_Count(environment, args.step_size, args.gamma, args.epsilon)
            time_step = 0
            counter = 0
            undiscounted_return = np.zeros(100000-2)
            while not environment.is_terminal():
                # undisc_return[time_step] = qlearning_joint_agent.get_undisc_return()
                qlearning_tile_count.step()
                if time_step < 100000-2:
                    undisc_return[time_step][run] = qlearning_tile_count.get_undisc_return()
                    for state in range(6):
                        for action in range(2):
                            bonus_rewards[state][action][time_step][run] = qlearning_tile_count.get_count_reward(state, action)

                # print(time_step, ",", qlearning_joint_agent.get_undisc_return())
                time_step += 1 

            environment.reset()

        
    # Visualization.plot_undis_return(undisc_return, "Tile Coding Intrinsic Reward", number_of_run)
    with open('results/tile_coding_count_bonus_min.npy', 'wb') as f:
        np.save(f, undisc_return)
    
    with open('results/tile_coding_count_bonus_min_bonus_rewards.npy', 'wb') as f:
        np.save(f, bonus_rewards)


