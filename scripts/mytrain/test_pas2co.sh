#!/bin/bash

#SBATCH --job-name=pa2co
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

SPLIT=$1
LAYERS=$2
SHOT=$3


echo "start"
singularity exec --nv \
            --overlay /scratch/cz1627/overlay-25GB-500K.ext3:ro \
			/scratch/cz1627/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
            /bin/bash -c " source /ext3/env.sh;
            python -m src.test_kshot --config config_files/pascal2coco.yaml \
					 --opts train_split ${SPLIT} \
					 		test_split ${SPLIT} \
						    layers ${LAYERS} \
						    shot ${SHOT} \
					 > log_pas2co.txt 2>&1"

echo "finish"


#GREENE GREENE_GPU_MPS=yes

