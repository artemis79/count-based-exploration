#!/bin/bash
#SBATCH --account=def-mbowling
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G      
#SBATCH --time=2:59:00
#SBATCH --output=qlearning.out

# SOCKS5 Proxy
if [ "$SLURM_TMPDIR" != "" ]; then
    echo "Setting up SOCKS5 proxy..."
    ssh -q -N -T -f -D 8888 `echo $SSH_CONNECTION | cut -d " " -f 3`
    export ALL_PROXY=socks5h://localhost:8888
fi

# Setup Modules
module load python/3.10

# Setup Python Environment
cd $SLURM_TMPDIR
python -m venv venv
source venv/bin/activate
echo "load python"
pip install --no-index numpy
pip install --no-index matplotlib
pip install --no-index seaborn
pip install --no-index tqdm
pip install --no-index wandb

#git clone
git config --global http.proxy 'socks5://127.0.0.1:8888'
git clone --quiet https://github.com/artemis79/count-based-exploration.git $SLURM_TMPDIR/project


# Run experiment
echo "Running experiment..."
cd $SLURM_TMPDIR/project
mkdir results

#wandb variables
export WANDB_API_KEY=c661d4027cae102ac37a9dd80433c1648bab0e56
export SWEEP_ID=$(wandb sweep tabular/config/qlearning_config.yaml --project tile-coding-exploration)
wandb agent --project "tile-coding-explroation"  $SWEEPID

