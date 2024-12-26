srun --gres=gpu:1 -n 32 --mem=100G  --time 24:00:00  --pty /bin/bash

conda activate vlr-lmms-eval
cd code/Reasoning_vlr/outputs/lmms-eval
