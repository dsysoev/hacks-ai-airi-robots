import datetime

import gym
from pogema import GridConfig
from pogema.animation import AnimationMonitor

from model import Model


def main(anim=False, seed=1, map_size='tiny', egocentric_idx=None):
    # Define random configuration
    if map_size == 'huge':
        grid_config = GridConfig(num_agents=64,  # количество агентов на карте
                                 size=32,  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=seed,  # сид генерации задания
                                 max_episode_steps=128,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
    elif map_size == 'medium':
        grid_config = GridConfig(num_agents=32,  # количество агентов на карте
                                 size=16,  # размеры карты
                                 density=0.2,  # плотность препятствий
                                 seed=seed,  # сид генерации задания
                                 max_episode_steps=64,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
    elif map_size == 'tiny':
        grid_config = GridConfig(num_agents=16,  # количество агентов на карте
                                 size=8,  # размеры карты
                                 density=0.1,  # плотность препятствий
                                 seed=seed,  # сид генерации задания
                                 max_episode_steps=32,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
    else:
        raise NotImplementedError()

    env = gym.make("Pogema-v0", grid_config=grid_config, disable_env_checker=True)
    if anim:
        env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver = Model()
    steps = 0

    start = datetime.datetime.now()
    prev_done_sum = 0
    reward = []
    while not all(done):
        prev_done_sum = sum(done)
        # Используем AStar
        obs, reward, done, info = env.step(solver.act(obs, done,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative()))
        steps += 1

    duration = (datetime.datetime.now() - start).seconds
    if anim:
        # сохраняем анимацию и рисуем ее
        env.save_animation("render.svg", egocentric_idx=egocentric_idx)

    # calc metrics
    isr = (prev_done_sum + sum(reward)) / len(done)
    metrics = {'steps': steps, 'ICR': isr, 'CSR': int(isr == 1.), 'duration': duration}

    return metrics


if __name__ == '__main__':
    print(main())
