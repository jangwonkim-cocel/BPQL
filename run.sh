# ----Test Environment----
# python == 3.8.10
# gym == 0.26.2
# mujoco_py == 2.1.2.14
# pytorch == 2.1.0
# numpy == 1.24.3
#-------------------------

#!/bin/bash
python main.py \
--env-name "HalfCheetah-v3" \
--random-seed 2023 \
--obs-delayed-steps 5 \
--act-delayed-steps 4 \
--max-step 1000000
