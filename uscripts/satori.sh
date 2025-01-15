srun --gres=gpu:1 -n 32 --mem=100G  --time 24:00:00  --pty /bin/bash
srun --gres=gpu:4 -n 32 --mem=250G  --time 24:00:00  --qos=sched_level_2 --pty /bin/bash 

conda activate vlr-lmms-eval
cd ~/code/Reasoning_vlr/outputs/lmms-eval

