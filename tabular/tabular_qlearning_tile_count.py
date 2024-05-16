import utils
import random
import numpy as np
import environment as env
from qlearning_tile_count import QLearning_Tile_Count
from utils import Visualization
from tqdm import tqdm

actions = ["right", "left"]

if __name__ == "__main__":
    # Read arguments:
    number_of_run = 50
    args = utils.ArgsParser.read_input_args()

    environment = env.MDP(args.input)

    undisc_return = np.zeros((100000-2, number_of_run))

    for run in tqdm(range(0, number_of_run)):
        random.seed(run)
        np.random.seed(run)
        for ep in range(args.num_episodes):
            qlearning_joint_agent= QLearning_Tile_Count(environment, args.step_size, args.gamma, args.epsilon)
            time_step = 0
            counter = 0
            undiscounted_return = np.zeros(100000-2)
            while not environment.is_terminal():
                # undisc_return[time_step] = qlearning_joint_agent.get_undisc_return()
                qlearning_joint_agent.step()
                if time_step < 100000-2:
                    undisc_return[time_step][run] = qlearning_joint_agent.get_undisc_return()

                # print(time_step, ",", qlearning_joint_agent.get_undisc_return())
                time_step += 1 

            environment.reset()

        
    Visualization.plot_undis_return(undisc_return, "Tile Coding Intrinsic Reward", number_of_run)
       



