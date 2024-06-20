
import utils
import random
import numpy as np
import environment as env
from qlearning_count import QLearning_Count
from tqdm import tqdm


actions = ["right", "left"]

if __name__ == "__main__":
    # Max timestep for each episode
    number_of_run = 100
    # Read arguments:
    args = utils.ArgsParser.read_input_args()

    # Instantiate objects I'll need
    environment = env.MDP(args.input)

    # Actual learning algorithm
    undisc_return = np.zeros((100000, number_of_run))
    for run in tqdm(range(0, number_of_run)):
        random.seed(run)
        np.random.seed(run)
        for ep in range(args.num_episodes):
            agent = QLearning_Count(environment, args.step_size, args.gamma, args.epsilon)
            time_step = 1
            while not environment.is_terminal():
                prev_state = agent.curr_s
                if time_step < 100000:
                    undisc_return[time_step][run] = agent.get_undisc_return()
                agent.step()
                time_step += 1
        environment.reset()

    with open('results/count_bonus.npy', 'wb') as f:
        np.save(f, undisc_return)


    

