# Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback
This repository contains the PyTorch implementation of **Belief Projection-Based Q-Learning (BPQL)** introduced in the paper:

**Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback** by Jangwon Kim et al., presented at Advances in Neural Information Processing Systems (NeurIPS), 2023.


## Paper Link
>See the paper here: https://proceedings.neurips.cc/paper_files/paper/2023/hash/0252a434b18962c94910c07cd9a7fecc-Abstract-Conference.html

## Test Environment
```
python == 3.8.10
gym == 0.26.2
mujoco_py == 2.1.2.14
pytorch == 2.1.0
numpy == 1.24.3
```

## How to Run?
### Run the script file 
```
>chmod +x run.sh
>./run.sh
```

### or run main.py with arguments
```
python main.py --env-name HalfCheetah-v3 --random-seed 2023 --obs-delayed-steps 5 --act-delayed-steps 4 --max-step 1000000
```
---

## Citation Example
```
@inproceedings{kim2023cocel,
   author = {Kim, Jangwon and Kim, Hangyeol and Kang, Jiwook and Baek, Jongchan and Han, Soohee},
   booktitle = {Advances in Neural Information Processing Systems},
   pages = {678--696},
   title = {Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback},
   volume = {36},
   year = {2023}
}
```
