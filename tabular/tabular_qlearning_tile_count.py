import wandb
import utils
import random
import numpy as np
import environment as env
from qlearning_tile_count import QLearning_Tile_Count
from tqdm import tqdm

actions = ["right", "left"]

if __name__ == "__main__":

    wandb.init(project="tile-coding-explroation")

    # Read arguments:
    args = utils.ArgsParser.read_input_args()
    num_seeds = (args.num_seeds)

    environment = env.MDP(args.input)
    time_limit = environment.get_time_limit()

    undisc_return = np.zeros((num_seeds, time_limit+1))
    bonus_rewards = np.zeros((6, 2, num_seeds ,time_limit+1))

    total_return = []


    for run in tqdm(range(0, num_seeds)):
        random.seed(run)
        np.random.seed(run)
        func = np.min
        if args.aggregate_function == "sum":
            func = np.sum
        elif args.aggregate_function == "max":
            func = np.max

        qlearning_tile_count= QLearning_Tile_Count(environment, args.step_size, args.gamma, args.epsilon, args.beta, func)
        time_step = 0
        counter = 0
        undiscounted_return = np.zeros(num_seeds)
        while not environment.is_terminal():
            # undisc_return[time_step] = qlearning_joint_agent.get_undisc_return()
            if time_step <= time_limit:
                qlearning_tile_count.step()
                undisc_return[run][time_step] = qlearning_tile_count.get_undisc_return()
                for state in range(6):
                    for action in range(2):
                        bonus_rewards[state][action][run][time_step] = qlearning_tile_count.get_count_reward(state, action)

            time_step += 1 

        total_return.append(undisc_return[run][time_limit-2])
        environment.reset()

    average_return = np.mean(total_return)
    print(average_return)
    wandb.log({'average_return': average_return})
    

    with open('results/tile_coding_count_bonus_min.npy', 'wb') as f:
        np.save(f, undisc_return)
    
    with open('results/tile_coding_count_bonus_min_bonus_rewards.npy', 'wb') as f:
        np.save(f, bonus_rewards)


