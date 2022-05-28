import datetime

import gym
from pogema import GridConfig
from pogema.animation import AnimationMonitor

from model import Model


def main(anim=False, seed=1, map_size='tiny', egocentric_idx=None):
    # Define random configuration
    if map_size == 'huge':
        grid_config = GridConfig(num_agents=32,  # количество агентов на карте
                                 size=64,  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=seed,  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
    elif map_size == 'hard':
        grid_config = GridConfig(num_agents=128,  # количество агентов на карте
                                 size=32,  # размеры карты
                                 density=0.2,  # плотность препятствий
                                 seed=seed,  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
                                 obs_radius=5,  # радиус обзора
                                 )
    elif map_size == 'tiny':
        grid_config = GridConfig(num_agents=4,  # количество агентов на карте
                                 size=8,  # размеры карты
                                 density=0.3,  # плотность препятствий
                                 seed=seed,  # сид генерации задания
                                 max_episode_steps=256,  # максимальная длина эпизода
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
    while not all(done):
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
    isr = sum(done) / len(done)
    metrics = {'steps': steps, 'ICR': isr, 'duration': duration}

    return metrics


if __name__ == '__main__':
    print(main())
