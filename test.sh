#!/bin/bash
#SBATCH --time=96:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=log3.txt
#SBATCH -p new
#SBATCH --exclude=hlt06



python generate3.py