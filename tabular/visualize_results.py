import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

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
    def plot_undis_return(undisc_reutrn, name, num_runs, file_name):
        average_undisc = np.average(undisc_reutrn, axis=0)
        std_undisc = np.std(undisc_reutrn, axis=0)
        confidence_interval = 1.96*std_undisc/np.sqrt(num_runs)
        plt.ylabel("Undiscounted\nReturn")
        plt.xlabel("Time step")
        plt.plot(average_undisc, label=name)
        x = np.linspace(0, len(average_undisc)-1, len(average_undisc))
        plt.fill_between(x, y1= average_undisc-confidence_interval, y2=average_undisc+confidence_interval, alpha=0.5)
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.savefig(file_name)
        

    @staticmethod
    def plot_bonus_rewards(bonus_rewards, name, num_runs, file_name):
        for action in range(2):
            average_undisc = np.average(bonus_rewards[action], axis=0)
            std_undisc = np.std(bonus_rewards[action], axis=0)
            confidence_interval = 1.96*std_undisc/np.sqrt(num_runs)
            plt.ylabel("Bonus\nReward")
            plt.xlabel("Time step")
            if action == 0:
                plt.plot(average_undisc, label="state="+str(state)+"action=right")
            else:
                plt.plot(average_undisc, label="state="+str(state)+"action=left")

            x = np.linspace(0, len(average_undisc)-1, len(average_undisc))
            plt.fill_between(x, y1= average_undisc-confidence_interval, y2=average_undisc+confidence_interval, alpha=0.5)
            plt.legend(loc="upper left")
            plt.grid(True)
        plt.savefig("plots/bonus_reward_state="+str(state))
        plt.clf()

if __name__ == "__main__":
    directory = 'results'

    # for filename in os.listdir(directory):
    #     f = os.path.join(directory, filename)
    #     if "tile_coding" in f:
    #         name, _, _ = f.partition('.')
    #         name = name.replace('_', ' ')
    #         name = name.replace('results/', '')
    #         undisc_return = np.load(f)
    #         Visualization.plot_undis_return(undisc_return, name, np.shape(undisc_return)[1], 'plots/tile_coding')

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        name, _, _ = f.partition('.')
        name = name.replace('_', ' ')
        name = name.replace('results/', '')
        if name == "tile coding count bonus min":
            undisc_return = np.load(f)
            print(np.shape(undisc_return))
            print(name)
            Visualization.plot_undis_return(undisc_return, name, np.shape(undisc_return)[0], 'plots/undiscounted_return')


    # for filename in os.listdir(directory):
    #     f = os.path.join(directory, filename)
    #     name, _, _ = f.partition('.')
    #     name = name.replace('_', ' ')
    #     name = name.replace('results/', '')
    #     if name == "tile coding count bonus min bonus rewards" :
    #         bonus_rewards = np.load(f)  
    #         for state in range(6):
    #             bonus_rewards_state = bonus_rewards[state]
    #             Visualization.plot_bonus_rewards(bonus_rewards_state, name, np.shape(bonus_rewards)[1], 'plots/bonus_rewards')




        