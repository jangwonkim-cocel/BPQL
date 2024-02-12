# Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback
>PyTorch implementation of Belief Projection-Based Q-Learning (BPQL).<br/>
>See the paper here: https://openreview.net/pdf?id=sq0m11cUMV

## Test Environment
>python == 3.8.10<br/>
>gym == 0.26.2<br/>
>mujoco_py == 2.1.2.14<br/>
>pytorch == 2.1.0<br/>
>numpy == 1.24.3<br/>

## How to Run?
### Run the script file 
>chmod +x run.sh<br/>
>./run.sh

### or run main.py with arguments
> python main.py --env-name HalfCheetah-v3 --random-seed 2023 --obs-delayed-steps 5 --act-delayed-steps 4 --max-step 1000000
