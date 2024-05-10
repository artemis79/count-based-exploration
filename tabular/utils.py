import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def argmax(vector):
    # This argmax breaks ties randomly
    return np.random.choice(np.flatnonzero(vector == vector.max()))

class Visualization:
    @staticmethod
    def highlight_cell(x,y, ax=None, **kwargs):
        rect = plt.Rectangle((x-.5, y-.5), 2,1, fill=False, **kwargs)
        ax = ax or plt.gca()
        ax.add_patch(rect)
        return rect
    
    @staticmethod
    def riverswim_state_visitation(state_action_count):
        state_action_count[:, [1, 0]] = state_action_count[:, [0, 1]]
        state_action_count = state_action_count.flatten().reshape(1, 12)
        # state_count = np.sum(state_action_count, axis=1)
        # state_count = state_count.reshape(1, 6)
        fig, ax = plt.subplots()
        im = ax.imshow(state_action_count)


        #draw grid borders around each grid
        for i in range(0, 12, 2):
            Visualization.highlight_cell(i, 0, color='white', linewidth=2)
        
        cbar = ax.figure.colorbar(im, orientation="horizontal",ax = ax)
        cbar.ax.set_ylabel("Color bar", va = "bottom")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    @staticmethod
    def plot_undis_return(undisc_reutrn, name):
        plt.ylabel("Undiscounted\nReturn")
        plt.xlabel("Time step")
        plt.plot(undisc_reutrn, label=name)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

class ArgsParser:
    """
    Read the user's input and parse the arguments properly. When returning args, each value is properly filled.
    Ideally one shouldn't have to read this function to access the proper arguments, but I postpone this.
    """

    @staticmethod
    def read_input_args():
        # Parse command line
        parser = argparse.ArgumentParser(
            description='Define algorithm\'s parameters.')

        parser.add_argument('-s', '--seed', type=int, default=1, help='Seed to be used in the code.')
        parser.add_argument('-i', '--input', type=str, default='tabular/mdps/riverswim.mdp',
                            help='File containing the MDP definition (default: tabular/mdps/riverswim.mdp).')
        parser.add_argument('-n', '--num_episodes', type=int, default=1,
                            help='For how many episodes we are going to learn.')
        parser.add_argument('-a', '--step_size', type=float, default=0.1,
                            help="Algorithm's step size. Alpha parameter in algorithms such as Sarsa.")
        parser.add_argument('-y', '--step_size_sr', type=float, default=0.1,
                            help="Step size to compute the SR with TD when using it in algorithms such as Sarsa.")
        parser.add_argument('-b', '--beta', type=float, default=1.0,
                            help="Real reward = Real reward + beta * Intrinsic Reward.")
        parser.add_argument('-g', '--gamma', type=float, default=0.99,
                            help='Gamma. Discount factor to be used by the algorithm.')
        parser.add_argument('-z ', '--gamma_sr', type=float, default=0.95,
                            help='Gamma value to compute the SR.')
        parser.add_argument('-e', '--epsilon', type=float, default=0.1,
                            help='Epsilon. This is the exploration parameter (trade-off).')
        parser.add_argument('-r', '--reward_structure', type=str, default="",
                            help="Valid values: 'dot-prod', 'diff', 'gamma-diff', 'norm' ")
        parser.add_argument('-d', '--divide', type=bool, default=False,
                            help="If true, the reward is equal to 1/reward_structure")

        args = parser.parse_args()

        return args
