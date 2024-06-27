import wandb
import utils
import random
import numpy as np
import environment as env
from qlearning_agent import QLearning
from tqdm import tqdm


if __name__ == "__main__":
    # Wandb setup:
    wandb.init(project="tile-coding-explroation", settings=wandb.Settings(start_method="fork"))

    # Read arguments:
    args = utils.ArgsParser.read_input_args()
    num_seeds = (args.num_seeds)

    # Instantiate objects I'll need
    environment = env.MDP(args.input)
    time_limit = environment.get_time_limit()

    undisc_return = np.zeros((num_seeds, time_limit+1))
    total_return = []
    
    for run in tqdm(range(num_seeds)):
        # Actual learning algorithm        
        agent = QLearning(environment, args.step_size, args.gamma, args.epsilon)
        time_step = 1

        random.seed(run)
        np.random.seed(run)

        while not environment.is_terminal():
            agent.step()
            undisc_return[run][time_step] = agent.get_undisc_return()
            time_step += 1
            # print(time_step, ",", agent.get_undisc_return())

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
    with open('results/qlearning.npy', 'wb') as f:
        np.save(f, undisc_return)