#!/bin/bash

#SBATCH --job-name=k1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=seg.out
#SBATCH --gres=gpu # How much gpu need, n is the number

module purge

DATA=$1
SPLIT=$2
LAYERS=$3
SHOT=$4



echo "start"
singularity exec --nv \
            --overlay /scratch/lg154/python36/python36.ext3:ro \
            --overlay /scratch/lg154/sseg/dataset/coco2014.sqf:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c " source /ext3/env.sh;
            python -m src.train_mmn --config config_files/${DATA}_mmn.yaml \
					 --opts train_split ${SPLIT} \
						    layers ${LAYERS} \
						    shot ${SHOT} \
						    cls_lr 0.1 \
						    batch_size 1 \
						    batch_size_val 1 \
					 > log.txt 2>&1"

echo "finish"


#GREENE GREENE_GPU_MPS=yes


