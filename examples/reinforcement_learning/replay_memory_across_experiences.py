from avalanche.training.strategies.reinforcement_learning.dqn import DQNStrategy, default_dqn_logger
from avalanche.models.dqn import MLPDeepQN
from avalanche.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator
from avalanche.training.plugins.replay import ReplayPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.evaluation.metrics.ram_usage import MaxRAM
import torch
from torch.optim import Adam



if __name__ == "__main__":
    device = torch.device('cpu')

    scenario = gym_benchmark_generator(
        ['CartPole-v1', 'CCartPole-v1'],
        n_parallel_envs=1, eval_envs=['CartPole-v1', 'CCartPole-v1'], n_experiences=4, 
        env_kwargs={'CCartPole-v1': dict(gravity=0.1, length=1., masscart=1000., force_mag=1.)})

    my_logger = EvaluationPlugin(
    *default_dqn_logger.metrics,
    # TODO: is memory usage constantly increasing with experience replay lying around?
    MaxRAM(),
    loggers=default_dqn_logger.loggers)

    # CartPole setting
    model = MLPDeepQN(input_size=4, hidden_size=1024,
                      n_actions=2, hidden_layers=2)
    print("Model", model)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # replay memory containing samples from different experiences 
    # NOTE: replay memory and replay plugin size must be the same with default storage policy
    replay_size = 1000
    replay_plugin = ReplayPlugin(mem_size=replay_size)

    strategy = DQNStrategy(
        model, optimizer, 100, batch_size=16, replay_memory_size=replay_size,
        updates_per_step=5, replay_memory_init_size=1000,
        target_net_update_interval=10, eval_every=-1, eval_episodes=10,
        plugins=[replay_plugin],
        device=device, evaluator=my_logger)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Task", experience.task_label, type(experience.task_label))
        strategy.train(experience)
        if experience.current_experience > 0:
            print("ReplayDataLoader", strategy.dataloader)
            for t in strategy.dataloader:
                print(t)
                break



    print('Training completed')
    eval_episodes = 100
    print(f"\nEvaluating on {eval_episodes} episodes!")
    print(strategy.eval(scenario.test_stream))
