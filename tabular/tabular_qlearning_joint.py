import utils
import random
import numpy as np
import environment as env
from qlearning_joint import QLearning_Joint
from utils import Visualization
from tqdm import tqdm

actions = ["right", "left"]

if __name__ == "__main__":
    # Read arguments:
    args = utils.ArgsParser.read_input_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    environment = env.MDP(args.input)

    undisc_return = np.zeros(100000)

    for ep in range(args.num_episodes):
        qlearning_joint_agent= QLearning_Joint(environment, args.step_size, args.gamma, args.epsilon)
        time_step = 0
        counter = 0
        undiscounted_return = np.zeros(100000-2)

        while not environment.is_terminal():
            # undisc_return[time_step] = qlearning_joint_agent.get_undisc_return()
            qlearning_joint_agent.step()
            if time_step < 100000-2:
                undisc_return[time_step] = qlearning_joint_agent.get_undisc_return()

            print(time_step, ",", qlearning_joint_agent.get_undisc_return())
            time_step += 1 

        environment.reset()

        print(qlearning_joint_agent.q)
        print(qlearning_joint_agent.q_count)
        print(qlearning_joint_agent.state_action_visitation_count)

        Visualization.riverswim_state_visitation(qlearning_joint_agent.state_action_visitation_count)
        undisc_return = undisc_return[:-100]
        # Visualization.plot_undis_return(undisc_return, "Separate Q_c and Q_r")
        # print(undisc_return)



