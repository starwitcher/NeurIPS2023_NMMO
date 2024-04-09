# NeurIPS2023_NMMO Competition Code

## 1. 代码结构
这里仅对修改的部分部分进行说明
- reinforcement_learning: 强化学习赛道的文件夹
  - config.py: 配置文件，修改了其中的seed和奖励参数，其余为修改
- environment.py: 修改了其中的reward_done_info()函数

## 2. 环境配置
- 操作系统：win10+wsl2+Ubuntu 20.04
- GPU：RTX 4090，英伟达驱动版本：531.79，CUDA版本：12.1.112
- python版本：python3.9
- pytorch版本：1.13.1+cu117
- 其余包的版本见requirements.txt，与baseline版本一致，

## 3. 提交说明
只对baseline的奖励部分进行了调整
- 以下修改起到了效果
  - 死亡奖励：我也不知道为什么，奖励agent死亡会取得更高的分数;
  - 健康奖励：血量增加奖励，反之惩罚。灵感来自于去年的冠军-realikun;
  - 任务奖励：增加了task_with_2_reward、task_with_0p2_max_reward和task_completed_reward奖励，以CompletedTaskCount为优化目标，这里很容易理解;
  - unique_event奖励：增加原有奖励的值，unique_event中可能有一些利于agent生存的课程，增加其奖励是有效的.
- 以下方法应该有效，但是没有时间测试了
  - 使用随机搜索的方式寻找奖励的最优值，这种方法是有用的。一开始我理解是优化TotalScore，经过50轮的随机优化，得到了更好的结果，但是没有来得及优化CompletedTaskCount，我觉得理论上是可行的;
  - 【重点】2个阶段的训练方式，优化TotalScore得到了更优的值(60->64)，原则就是一阶段重奖励和生存，二阶段重惩罚（存疑）和成绩，这里的成绩可以是TotalScore或者CompletedTaskCount,
  - 例如在一阶段explore_bonus_weight==0.5,clip_unique_event==3，而二阶段分别为1,6，效果确实更好，同时也需要配合其他的值.
- 还尝试了以下的方法，但是没有效果
  - 使用奖励衰减，也就是reward=now_reward - average_last_epoch_reward，然而并没有什么用;
  - 增加吃血瓶奖励，但貌似agent从来没有学会吃血瓶;
  - 增加吃食物或者水奖励，负面效果或者几乎没有用;
  - 增加生存奖励，每生存100回合，给予奖励，负面效果或者几乎没有用;
  - 如果因为食物和水不足，而扣除健康，则进行惩罚，负面效果或者几乎没有用;
  - 增加死亡惩罚，没有效果，反而是需要在死亡的时候进行奖励，很奇怪。
- 相关的训练曲线可以在wandb中查看，其中"The optimal value in the reinforcement learning track"表示本次强化学习的最优值
  - https://wandb.ai/mori42/nips2023-nmmo?workspace=user-mori42

## 4. 运行与启动
- 我使用pytorch作为编辑器，在其中直接运行代码，使用下面的代码也能够运行
- python train.py

## 5. 环境存在的问题
- 环境随机性过强，即使参数配置正确，也需要测试多个seed才能看到效果，这与一开始限制算力的目标不一致;
- baseline缺少注释，看了半个月才大致看懂;
- trainer.close()貌似关闭不完全，不能直接使用wandb进行超参数搜索，只能每次手动配置参数启动.
