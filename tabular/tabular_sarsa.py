import utils
import random
import numpy as np
import environment as env
from sarsa_agent import Sarsa
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
    for ep in range(args.num_episodes):
        agent = Sarsa(environment, args.step_size, args.gamma, 0.5)
        time_step = 1
        while not environment.is_terminal():
            agent.step()
            time_step += 1
            print(time_step, ",", agent.get_undisc_return())
            if agent.curr_s == 3:
                counter += 1
        environment.reset()

    print(agent.state_action_visitation_count)
    Visualization.riverswim_state_visitation(agent.state_action_visitation_count)

