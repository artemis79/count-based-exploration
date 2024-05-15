#!/bin/bash
#SBATCH --account=def-mbowling
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G       
#SBATCH --time=0-0:10
#SBATCH --signal=USR1@300
#SBATCH --gres=gpu:1

# SOCKS5 Proxy
if [ "$SLURM_TMPDIR" != "" ]; then
    echo "Setting up SOCKS5 proxy..."
    ssh -q -N -T -f -D 8888 `echo $SSH_CONNECTION | cut -d " " -f 3`
    export ALL_PROXY=socks5h://localhost:8888
fi

# Setup Modules
module load python/3.10.2 gcc/11.3.0 cuda/11.8.0

# Setup Python Environment
cd $SLURM_TMPDIR
virtualenv pyenv
. pyenv/bin/activate