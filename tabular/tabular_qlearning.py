import utils
import random
import numpy as np
import environment as env
from qlearning_agent import QLearning
from tqdm import tqdm
from utils import Visualization

actions = ["right", "left"]

if __name__ == "__main__":
    # Read arguments:
    args = utils.ArgsParser.read_input_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Instantiate objects I'll need
    environment = env.MDP(args.input)

    # Actual learning algorithm
    counter = 0
    undisc_return = np.zeros(100000)
    for ep in range(args.num_episodes):
        agent = QLearning(environment, args.step_size, args.gamma, args.epsilon)
        time_step = 1
        while not environment.is_terminal():
            agent.step()
            if time_step < 100000:
                undisc_return[time_step] = agent.get_undisc_return()
            time_step += 1
            print(time_step, ",", agent.get_undisc_return())
            if agent.curr_s == 3:
                counter += 1
        environment.reset()

    print(agent.state_action_visitation_count)
    Visualization.riverswim_state_visitation(agent.state_action_visitation_count)
    # Visualization.plot_undis_return(undisc_return, "epsilong-greedy")

