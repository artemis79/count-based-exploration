
import utils
import random
import numpy as np
import environment as env
from qlearning_count_agent import QLearning_Count
from qlearning_count_as_reward import QLearning_Count_Reward
from tqdm import tqdm
from utils import Visualization

actions = ["right", "left"]

if __name__ == "__main__":
    # Max timestep for each episode
    max_time_step = 5000
    # Read arguments:
    args = utils.ArgsParser.read_input_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Instantiate objects I'll need
    environment = env.MDP(args.input)

    # Actual learning algorithm
    undisc_return = np.zeros(100000)
    for ep in range(args.num_episodes):
        agent = QLearning_Count(environment, args.step_size, args.gamma, args.epsilon)
        time_step = 1
        while not environment.is_terminal():
            prev_state = agent.curr_s
            if time_step < 100000:
                undisc_return[time_step] = agent.get_undisc_return()
            agent.step()
            time_step += 1
            # print(time_step, ",", agent.get_undisc_return())
            
        
        print(agent.state_action_visitation_count)
        Visualization.riverswim_state_visitation(agent.state_action_visitation_count)    
        # Visualization.plot_undis_return(undisc_return, "Intrinsic Reward")
        environment.reset()
