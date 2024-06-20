import wandb

wandb.login()

# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score

# 2: Define the search space
sweep_configuration = {
   "program": "tabular/tabular_qlearning.py",
   "name": "qlearning",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "average_return"},
    "parameters": {
        "step_size": {"values": [0.0001, 0.001, 0.01, 0.1]},

        "epsilon": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "gamma": {"value": 0.99},
        "num_seeds": {"value": 50}
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="tile-coding-explroation")

wandb.agent(sweep_id, project="tile-coding-explroation")