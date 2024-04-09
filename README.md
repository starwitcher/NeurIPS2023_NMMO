# NeurIPS2023_NMMO Competition Code

## 1. Code Structure
This section only explains the modifications made.
- reinforcement_learning: Folder for the reinforcement learning track
  - config.py: Configuration file, modified the seed and reward parameters, the rest remained unchanged
- environment.py: Modified the reward_done_info() function

## 2. 环境配置
- Operating system: Windows 10 + WSL2 + Ubuntu 20.04
- GPU: RTX 4090, NVIDIA driver version: 531.79, CUDA version: 12.1.112
- Python version: Python 3.9
- PyTorch version: 1.13.1+cu117
- The versions of other packages are consistent with the baseline as per the requirements.txt file.

## 3. 提交说明
Only the reward section of the baseline was adjusted.
- The following modifications were effective:
  - Death reward: I'm not sure why, but rewarding agent death has resulted in higher scores;
  - Health reward: Incentivized health gains, penalized health losses. This was inspired by last year's champion, realikun;
  - Task rewards: Added task_with_2_reward, task_with_0p2_max_reward, and task_completed_reward rewards, optimizing for CompletedTaskCount. This is straightforward to understand;
  - Unique event reward: Increased the value of unique_event rewards, as some unique events may be beneficial to agent survival. Boosting this reward has been effective.
- The following methods should be effective but were not tested due to time constraints:
  - Using random search to find the optimal value of the reward is useful. Initially, I understood that the goal was to optimize TotalScore, and after 50 rounds of random optimization, I obtained better results, but I didn't have time to optimize CompletedTaskCount. I think it is theoretically feasible.
  - 2-phase training approach, optimizing TotalScore in the first phase and focusing on punishment (with some reservation) and performance in the second phase. Performance can be measured by TotalScore or CompletedTaskCount. For example, setting explore_bonus_weight to 0.5 and clip_unique_event to 3 in the first phase and adjusting them to 1 and 6 in the second phase has worked better;
- Also attempted the following methods but they were ineffective:
  - Using reward decay by subtracting the average last epoch reward from the current reward. This had no effect;
  - Incentivizing consumption of blood potions but the agent never learned to do so;
  - Incentivizing consumption of food or water but this had a negative effect or was ineffective;
  - Penalizing survival but this had a negative effect or was ineffective; 
  - if food and water are scarce and the agent's health is reduced due to lack of consumption, penalizing it was ineffective;
  - Penalizing deaths but this had no effect - it would be more effective to reward death in certain scenarios which is counterintuitive.
- The relevant training curves can be viewed on WandB, specifically "The optimal value in the reinforcement learning track" refers to the optimal value of the reinforcement learning track for this submission.
  - https://wandb.ai/mori42/nips2023-nmmo?workspace=user-mori42

## 4. ## Running and Starting
- I used PyTorch as the editor and ran the code directly within it. The following command can also be used to run the code:
- python train.py

## 5. Environmental Issues
- The environment has strong randomness even with correct parameter configuration, multiple seeds need to be tested to see the actual effect, which is inconsistent with the initial goal of limiting computational resources;
- The baseline lacks comments and took me about half a month to roughly understand it;
- The trainer's close method does not seem to close completely, WandB cannot be used for hyperparameter search directly, manual configuration and launch is required each time.
