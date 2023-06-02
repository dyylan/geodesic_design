#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --ntasks-per-core 1
#SBATCH --qos=normal
#SBATCH --account=vector
#SBATCH --output=outputs/slurm-%A_%a.out
#SBATCH --mem=4000M        # memory per node
#SBATCH --time=3:00:00
#SBATCH --job-name=unitary_design
#SBATCH --array=[1-100]

export PATH=/pkgs/anaconda3/bin:$PATH
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /h/roeland/condaenvs/unitary_design/
echo $1
python -u cluster.py --gate=$1 --instance=$SLURM_ARRAY_TASK_ID
wait
