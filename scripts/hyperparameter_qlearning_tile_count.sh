#!/bin/bash
#SBATCH --account=def-mbowling
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G      
#SBATCH --time=2:59:00
#SBATCH --output=outputs/qlearning_tile_count_%j.out
#SBATCH --array=1-50

# Setup Modules
module load python/3.10
scp count-based-exploration.tar.gz $SLURM_TMPDIR

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



tar -xvzf count-based-exploration.tar.gz


#git clone
#ssh -q -N -T -f -D 8888 `echo $SSH_CONNECTION | cut -d " " -f 3`
#export ALL_PROXY=socks5h://localhost:8888
#git config --global http.proxy 'socks5://127.0.0.1:8888'
#git clone --quiet https://github.com/artemis79/count-based-exploration.git $SLURM_TMPDIR/project


# Run experiment
echo "Running experiment..."
cd $SLURM_TMPDIR/count-based-exploration
mkdir results

#wandb variables
wandb agent university-alberta/tile-coding-exploration/gndg2ed7


