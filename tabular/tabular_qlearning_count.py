import wandb
import utils
import random
import numpy as np
import environment as env
from qlearning_count import QLearning_Count
from tqdm import tqdm


actions = ["right", "left"]

if __name__ == "__main__":
    wandb.init()

    # Read arguments:
    args = utils.ArgsParser.read_input_args()
    num_seeds = args.num_seeds

    # Instantiate objects I'll need
    environment = env.MDP(args.input)
    time_limit = environment.get_time_limit()
    undisc_return = np.zeros((num_seeds, time_limit+1))
    total_return = []
    
    # Actual learning algorithm
    for run in tqdm(range(0, num_seeds)):
        random.seed(run)
        np.random.seed(run)

        agent = QLearning_Count(environment, args.step_size, args.gamma, args.epsilon, args.beta)
        time_step = 1

        while not environment.is_terminal():
            prev_state = agent.curr_s
            undisc_return[run][time_step] = agent.get_undisc_return()
            agent.step()

        total_return.append(agent.get_avg_undisc_return())
        environment.reset()

    average_return = np.mean(total_return)
    std_return = np.std(total_return)
    max_return = np.max(total_return)
    min_return = np.min(total_return)

    wandb.log({'average_return': average_return,
               'std_return': std_return,
               'min_return': min_return,
               'max_return': max_return})
    with open('results/count_bonus.npy', 'wb') as f:
        np.save(f, undisc_return)


    

