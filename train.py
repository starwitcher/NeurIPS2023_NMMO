import os
import logging
import torch

from pufferlib.vectorization import Serial, Multiprocessing
from pufferlib.policy_store import DirectoryPolicyStore
from pufferlib.frameworks import cleanrl
import wandb
import environment
from collections import defaultdict
from dataclasses import asdict
import numpy as np

os.environ["WANDB_API_KEY"] = '069b0ec3972c3b3e6e8133986357df933e7f9309' # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"   # 离线  （此行代码不用修改）

from reinforcement_learning import clean_pufferl, policy, config

# NOTE: this file changes when running curriculum generation track
# Run test_task_encoder.py to regenerate this file (or get it from the repo)
BASELINE_CURRICULUM_FILE = "reinforcement_learning/curriculum_with_embedding.pkl"
CUSTOM_CURRICULUM_FILE = "curriculum_generation/custom_curriculum_with_embedding.pkl"

def setup_env(args):
    run_dir = os.path.join(args.runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logging.info("Training run: %s (%s)", args.run_name, run_dir)
    logging.info("Training args: %s", args)

    policy_store = None
    if args.policy_store_dir is None:
        args.policy_store_dir = os.path.join(run_dir, "policy_store")
        logging.info("Using policy store from %s", args.policy_store_dir)
        policy_store = DirectoryPolicyStore(args.policy_store_dir)

    def make_policy(envs):
        learner_policy = policy.Baseline(
            envs.driver_env,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            task_size=args.task_size
        )
        return cleanrl.Policy(learner_policy)

    trainer = clean_pufferl.CleanPuffeRL(
        device=torch.device(args.device),
        seed=args.seed,
        env_creator=environment.make_env_creator(args),
        env_creator_kwargs={},
        agent_creator=make_policy,
        data_dir=run_dir,
        exp_name=args.run_name,
        policy_store=policy_store,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_extra_data=args,
        checkpoint_interval=args.checkpoint_interval,
        vectorization=Serial if args.use_serial_vecenv else Multiprocessing,
        total_timesteps=args.train_num_steps,
        num_envs=args.num_envs,
        num_cores=args.num_cores or args.num_envs,
        num_buffers=args.num_buffers,
        batch_size=args.rollout_batch_size,
        learning_rate=args.ppo_learning_rate,
        selfplay_learner_weight=args.learner_weight,
        selfplay_num_policies=args.max_opponent_policies + 1,
        #record_loss = args.record_loss,
    )
    return trainer

def rl_train():
    # 系统参数
    args = config.create_config(config.Config)
    args.tasks_path = BASELINE_CURRICULUM_FILE
    # 传递参数
    wandb.init(
        project=args.wandb_project,
        # config = args,
        # id = args.wandb_id,
        # resume = True,
    )
    # print(wandb.config)
    # print(wandb.config.hidden_size)
    # args.seed = wandb.config.seed
    # args.hidden_size = wandb.config.hidden_size
    # args.task_size = wandb.config.task_size
    # args.ppo_learning_rate = wandb.config.ppo_learning_rate
    args.heal_bonus_weight = wandb.config.heal_bonus_weight
    args.meander_bonus_weight = wandb.config.meander_bonus_weight
    args.explore_bonus_weight = wandb.config.explore_bonus_weight
    # 生成环境
    trainer = setup_env(args)
    results = defaultdict(list)
    while not trainer.done_training():
        _, stats, infos = trainer.evaluate()
        entropy_loss = trainer.train(
            update_epochs=args.ppo_update_epochs,
            bptt_horizon=args.bptt_horizon,
            batch_rows=args.ppo_training_batch_size // args.bptt_horizon,
            clip_coef=args.clip_coef,
        )
        for pol, vals in infos.items():
            results[pol].extend([
                e[1] for e in infos[pol]['team_results']
            ])
        total_score = rl_eval_epoch(results, args)
        # wandb.log(
        #     {
        #         "total_score": total_score,
        #     }
        # )
        # 如果熵小于一定的值，则break
        if entropy_loss < 1:
            print('entropy_loss<1')
            break
    trainer.close()


def rl_train_epoch(args, trainer):
    results = defaultdict(list)
    while not trainer.done_training():
        _, stats, infos = trainer.evaluate()
        for pol, vals in infos.items():
            results[pol].extend([
                e[1] for e in infos[pol]['team_results']
            ])
        entropy_loss = trainer.train(
            update_epochs=args.ppo_update_epochs,
            bptt_horizon=args.bptt_horizon,
            batch_rows=args.ppo_training_batch_size // args.bptt_horizon,
            clip_coef=args.clip_coef,
        )
        # 如果熵小于一定的值，则break
        if entropy_loss < 1:
            break

    return results


def rl_eval_epoch(results, args):
    # 返回获得的分数
    for pol, res in results.items():
        aggregated = {}
        keys = asdict(res[0]).keys()
        for k in keys:
            if k == 'policy_id':
                continue
            aggregated[k] = np.mean([asdict(e)[k] for e in res])
        results[pol] = aggregated
    # print(results['learner']['total_score'])
    total_score = int(results['learner']['total_score'])
    return total_score


def reinforcement_learning_track_auto_tuning_parameters(args):

    sweep_config = {
        'method': 'random',  # bayes
        'parameters': {
            # 'seed': {'value:': [130, 260, 700]},
            # 'hidden_size': {'values': [128, 256, 512, 1024]},
            # 'task_size': {'values': [2048, 4096, 8192]},
            # 'ppo_learning_rate': {'values': [0.00075, 0.00015, 0.00003]},
            'heal_bonus_weight': {'values': [0.001, 0.005, 0.01, 0.03, 0.05, 0.09, 0.12, 0.15, 0.2, 0.25]},
            'meander_bonus_weight': {'values': [0.001, 0.005, 0.01, 0.02, 0.05, 0.09, 0.12, 0.15, 0.2, 0.25]},
            'explore_bonus_weight': {'values': [0.001, 0.005, 0.01, 0.03, 0.05, 0.09, 0.12, 0.15, 0.2, 0.25]}
        }
    }
    metric = {
        'name': 'total_score',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric

    # wandb.init(project=args.wandb_project, entity=args.wandb_entity, config={'seed', 'hidden_size', 'task_size', 'ppo_learning_rate', 'heal_bonus_weight', 'meander_bonus_weight', 'explore_bonus_weight'})
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    # wandb.init()
    wandb.agent(sweep_id, function=rl_train, count=100)


def reinforcement_learning_track(trainer, args):
    while not trainer.done_training():
        trainer.evaluate()
        trainer.train(
            update_epochs=args.ppo_update_epochs,
            bptt_horizon=args.bptt_horizon,
            batch_rows=args.ppo_training_batch_size // args.bptt_horizon,
            clip_coef=args.clip_coef,
        )


def curriculum_generation_track(trainer, args, use_elm=True):
    from curriculum_generation.task_encoder import TaskEncoder
    LLM_CHECKPOINT = "Salesforce/codegen25-7b-instruct"

    if use_elm:
        from curriculum_generation import manual_curriculum
        from curriculum_generation.elm import OpenELMTaskGenerator
        AGENT_MODEL_PATH = ""
        NUM_SEED_TASKS = 20
        NUM_NEW_TASKS = 5
        ELM_DEBUG = True

        task_encoder = TaskEncoder(LLM_CHECKPOINT, manual_curriculum, batch_size=2)
        task_generator = OpenELMTaskGenerator(manual_curriculum.curriculum, LLM_CHECKPOINT)

        # @daveey: We need a baseline checkpoint for this
        #load_agent_model(AGENT_MODEL_PATH)

        # Generating new tasks and evaluating all candidate training tasks
        for _ in range(3):
            # NOTE: adjust NUM_SEED_TASKS to fit your gpu 调整NUM_SEED_TASKS以适合您的gpu
            seed_task_list = task_generator.sample_tasks(NUM_SEED_TASKS, random_ratio=1)
            new_task_list = task_generator.evolve_tasks(seed_task_list, NUM_NEW_TASKS, debug=ELM_DEBUG)
            task_generator.add_tasks(new_task_list)
            task_encoder.get_task_embedding(seed_task_list + new_task_list, save_to_file=CUSTOM_CURRICULUM_FILE)
            # CHECK ME: the trainer will automatically use the new task embedding file
            _, _, infos = trainer.evaluate()
            task_generator.update(infos) # update the task stats

        # NOTE: sample_tasks() uses task stats to sample learnable tasks
        curriculum = task_generator.sample_tasks(NUM_SEED_TASKS*3, random_ratio=0.3) # NOTE: arbitrary numbers

    else:
        from curriculum_generation import curriculum_tutorial  # custom tutorial
        task_encoder = TaskEncoder(LLM_CHECKPOINT, curriculum_tutorial, batch_size=2)
        curriculum = curriculum_tutorial.curriculum

    # Use the train_task_spec to train agents
    task_encoder.get_task_embedding(curriculum, save_to_file=CUSTOM_CURRICULUM_FILE)
    task_encoder.close()
    trainer.data.sort_keys = []
    reinforcement_learning_track(trainer, args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # You can either edit the defaults in config.py or set args
    # from the commandline.
    args = config.create_config(config.Config)

    if args.track == "rl":
      args.tasks_path = BASELINE_CURRICULUM_FILE
      trainer = setup_env(args)
      reinforcement_learning_track(trainer, args)
    elif args.track == "curriculum":
      args.tasks_path = CUSTOM_CURRICULUM_FILE
      trainer = setup_env(args)
      curriculum_generation_track(trainer, args, use_elm=True)
    else:
      raise ValueError(f"Unknown track {args.track}, must be 'rl' or 'curriculum'")

    trainer.close()
