#!/bin/bash

#SBATCH --job-name=1s_co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=cz1627@nyu.edu
#SBATCH --output=seg.out
#SBATCH --gres=gpu # How much gpu need, n is the number
#SBATCH --partition=a100,rtx8000



module purge

DATA=$1
SPLIT=$2
LAYERS=$3
SHOT=$4



echo "start"
singularity exec --nv \
            --overlay /scratch/cz1627/overlay-25GB-500K.ext3:ro \
			/scratch/cz1627/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
            /bin/bash -c " source /ext3/env.sh;
            python -m src.train_kshot --config config_files/${DATA}_mmn_check.yaml \
					 --opts train_split ${SPLIT} \
						    layers ${LAYERS} \
						    shot ${SHOT} \
					 > log_1s_coco.txt 2>&1"

echo "finish"


#GREENE GREENE_GPU_MPS=yes

