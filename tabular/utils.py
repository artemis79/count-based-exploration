import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def argmax(vector):
    # This argmax breaks ties randomly
    return np.random.choice(np.flatnonzero(vector == vector.max()))


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
        parser.add_argument('-ns', '--num_seeds', type=int, default=1, help='Number of seeds to run')
        parser.add_argument('-i', '--input', type=str, default='tabular/mdps/riverswim.mdp',
                            help='File containing the MDP definition (default: tabular/mdps/riverswim.mdp).')
        parser.add_argument('-t', '--time_step_limit', type=int, default=100000,
                            help="Number of time steps the agent runs in one run")
        parser.add_argument('-a', '--step_size', type=float, default=0.1,
                            help="Algorithm's step size. Alpha parameter in algorithms such as Sarsa.")
        parser.add_argument('-b', '--beta', type=float, default=1.0,
                            help="Real reward = Real reward + beta * Intrinsic Reward.")
        parser.add_argument('-g', '--gamma', type=float, default=0.99,
                            help='Gamma. Discount factor to be used by the algorithm.')
        parser.add_argument('-e', '--epsilon', type=float, default=0.1,
                            help='Epsilon. This is the exploration parameter (trade-off).')
        parser.add_argument('-r', '--reward_structure', type=str, default="",
                            help="Valid values: 'dot-prod', 'diff', 'gamma-diff', 'norm' ")
        parser.add_argument('-d', '--divide', type=bool, default=False,
                            help="If true, the reward is equal to 1/reward_structure")

        args = parser.parse_args()

        return args
    
