import numpy as np
import os
from agents.td3 import TD3
from agents.poly_td3_l0 import TD3 as polyTD3_l0
from agents.td3_no_param import TD3 as TD3_no_param
from utils import ReplayBuffer as TrainingBuffer
import torch
from envs.ks import KuramotoSivashinskyEnv
import matplotlib.pyplot as plt
import time
import argparse
def eval_agent(agent, eval_env, eval_episodes=1, idx=0, name='id_testing', best_rew=-1e6, agent_type='td3'):
    avg_reward = 0.
    avg_state_cost = 0.
    avg_action_cost = 0.
    best_policy = False
    start_time = time.time()

    for i in range(eval_episodes):
        steps = 0
        episode_state_cost = 0.
        episode_action_cost = 0.
        state_list = []
        action_list = []
        state, done = eval_env.reset(i=i), False
        while not done:
            action = agent.select_action(state)
            state_list.append(np.array(state))
            action_list.append(action)
            state, reward, done, errors = eval_env.step(action)
            avg_reward += reward
            avg_state_cost += errors[0]
            avg_action_cost += errors[1]
            episode_state_cost += errors[0]
            episode_action_cost += errors[1]
            steps += 1

        eval_env.render(name=name, idx=idx, best_policy=best_policy, agent_type=agent_type, i=i,
                        cost=[episode_state_cost, episode_action_cost])

    avg_reward /= eval_episodes
    avg_state_cost /= eval_episodes
    avg_action_cost /= eval_episodes
    if avg_reward > best_rew:
        best_rew = avg_reward
        best_policy = True
        eval_env.render(name=name, idx=idx, best_policy=best_policy, agent_type=agent_type,
                        cost=[avg_state_cost, avg_action_cost])

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} steps: {steps} Time: {np.round(time.time() - start_time, 2)}")
    print("---------------------------------------")
    return avg_reward, best_rew, [avg_state_cost, avg_action_cost]
def train_agent(agent, env, eval_env, od_eval_env, max_episodes, max_steps, warmup, eval_int, max_action, log,
                agent_type, save_int):
    count = 0
    best_id_rew = -1e6
    best_od_rew = -1e6
    for episode in range(max_episodes):
        start_time = time.time()
        state = env.reset()
        episode_reward = 0.0
        episode_state_cost = 0.0
        episode_action_cost = 0.0
        for step in range(max_steps):
            # Select action randomly or according to policy
            if episode < warmup:
                action = env.action_space.sample()
            else:
                action = (agent.select_action(state) + np.random.normal(0, max_action * 0.1, size=action_dim)).clip(-max_action, max_action)

            # perform action
            next_state, reward, done, errors = env.step(action)
            if done:
                done_bool = 1
            else:
                done_bool = 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward
            episode_state_cost += errors[0]
            episode_action_cost += errors[1]

            count += 1

            if done:
                # Train agent after collecting sufficient data
                if count > batch_size:
                    agent.train(replay_buffer, iterations=100, batch_size=batch_size)
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Episode Num: {episode + 1} Step: {step+1} Reward: {episode_reward:.3f} Time: {np.round(time.time() - start_time, 2)}")

                episode_reward = 0
                episode_state_cost = 0
                episode_action_cost = 0

                if episode % eval_int == 0 and episode != 0:

                    eval_rew, best_id_rew, id_errors = eval_agent(agent=agent, eval_episodes=6, eval_env=eval_env, idx=episode,
                                                       name='id_testing', best_rew=best_id_rew, agent_type=agent_type)
                    od_eval_rew, best_od_rew, od_errors = eval_agent(agent=agent, eval_episodes=8, eval_env=od_eval_env,
                                                          idx=episode, name='od_testing', best_rew=best_od_rew,
                                                          agent_type=agent_type)

                if episode % save_int == 0 and episode != 0:
                    directory = 'saved_models/' + agent_type + 'seed_' + str(seed)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    agent.save(filename='/' + agent_type + '_' + str(episode), directory=directory)

    eval_rew, _, _ = eval_agent(agent=agent, eval_episodes=4, eval_env=eval_env, idx=max_episodes + 1, name='id_testing',
                             agent_type=agent_type)
    od_eval_rew, _, _ = eval_agent(agent=agent, eval_episodes=2, eval_env=od_eval_env, idx=max_episodes + 1,
                                name='od_testing', agent_type=agent_type)

    directory = 'saved_models/' + agent_type + 'seed_' + str(seed)
    if not os.path.exists(directory):
        os.makedirs(directory)

    agent.save(filename='/' + agent_type + '_last', directory=directory)

def plot(rewards, w=10, name='reward'):
    plt.figure()
    plt.title(name)
    plt.plot(rewards)
    plt.plot(moving_average(rewards, w))
    plt.savefig("figures/" + name + ".png")
    plt.close()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-type', type=str, default='poly_td3_l0',
                        help='Agent type (td3, poly_td3_l0, td3_noparam.')
    parser.add_argument('--max-episodes', type=int, default=10000,
                        help='Number of training episodes.')
    parser.add_argument('--T', type=int, default=300,
                        help='Simulation time (seconds).')
    parser.add_argument('--Nx', type=int, default=64,
                        help='Spatial discretization of the Kuramoto-Sivashinshy PDE.')
    parser.add_argument('--Lx', type=int, default=22,
                        help='Domain size.')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Timestep PDE solver (seconds).')
    parser.add_argument('--frameskip', type=int, default=2,
                        help='Number of timesteps the control action is repeated.')
    parser.add_argument('--control-start', type=int, default=100,
                        help='Time at which the controller starts (seconds).')
    parser.add_argument('--oversampling', type=int, default=15,
                        help='Parameter PDE solver.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--h-dim', type=int, default=256,
                        help='Size hidden layers.')
    parser.add_argument('--warmup', type=int, default=500,
                        help='Number of episodes where a random policy is used.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--eval-int', type=int, default=1,
                        help='Policy evaluation interval (episodes).')
    parser.add_argument('--save-int', type=int, default=1000,
                        help='Saving the networks (episodes).')
    parser.add_argument('--nr-actuators', type=int, default=8,
                        help='Number of actuators.')
    parser.add_argument('--nr-sensors', type=int, default=8,
                        help='Number of sensors.')
    parser.add_argument('--offset', type=int, default=4,
                        help='Offset for position of sensors and actuators.')
    parser.add_argument('--action-scale', type=float, default=1.0,
                        help='Scale factor of the actions.')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Scaling parameter for reward function.')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Update rate target networks.')
    parser.add_argument('--mu', type=float, default=0.0,
                        help='Parameter of the system.')
    parser.add_argument('--sigma', type=float, default=0.8,
                        help='Std Gaussian actuators.')
    parser.add_argument('--param-dim', type=int, default=1,
                        help='Dimension parameter mu.')
    parser.add_argument('--parametric', type=bool, default=False,
                        help='Random mu at each training episodes (overwrites value of mu set).')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='Std Gaussian actuators.')
    parser.add_argument('--log', type=bool, default=False,
                        help='Log data (wandb).')
    parser.add_argument('--max-degree', type=int, default=3,
                        help='Maximum degree polynomial a-posteriori approximator.')
    parser.add_argument('--poly-degree', type=int, default=3,
                        help='Degree of the polynomial representing the policy (poly_td3_l0).')
    parser.add_argument('--sparsity-coeff', type=float, default=0.0005,
                        help='Coefficient for sparsity regularization.')
    parser.add_argument('--trunc-threshold', type=float, default=0.2,
                        help='Truncation threshold for poly_td3_l1_trunc.')
    args = parser.parse_args()

    # Simulation parameters
    agent_type = args.agent_type
    max_episodes = args.max_episodes
    T = args.T
    Nx = args.Nx
    Lx = args.Lx
    dt = args.dt
    frameskip = args.frameskip
    control_start = args.control_start
    max_steps = int(((T / dt) - (control_start / dt)) / frameskip) - 1
    oversampling = args.oversampling
    batch_size = args.batch_size
    h_dim = args.h_dim
    warmup = args.warmup
    seed = args.seed
    eval_int = args.eval_int
    nr_actuators = args.nr_actuators
    nr_sensors = args.nr_sensors
    action_scale = args.action_scale
    alpha = args.alpha
    tau = args.tau
    mu = args.mu
    sigma = args.sigma
    offset = args.offset
    param_dim = args.param_dim
    log = args.log
    parametric = args.parametric
    save_int = args.save_int
    action_scale = args.action_scale
    max_degree = args.max_degree
    poly_degree = args.poly_degree
    reg_coeff = args.sparsity_coeff
    trunc_threshold = args.trunc_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = 'KuramotoSivashinsky'

    env = KuramotoSivashinskyEnv(Nx=Nx, Lx=Lx, dt=dt, T=T, frameskip=frameskip, max_rl_steps=max_steps,
                                 parametric=parametric, action_scale=action_scale, nr_actuators=nr_actuators,
                                 nr_sensors=nr_sensors, oversampling=oversampling, alpha=alpha, mu=mu, sigma=sigma,
                                 offset=offset, control_start=control_start, seed=seed, eval=False)
    eval_env = KuramotoSivashinskyEnv(Nx=Nx, Lx=Lx, dt=dt, T=T, frameskip=frameskip, max_rl_steps=max_steps,
                                      parametric=parametric, action_scale=action_scale, nr_actuators=nr_actuators,
                                      nr_sensors=nr_sensors, oversampling=oversampling, alpha=alpha, mu=mu, sigma=sigma,
                                      offset=offset, control_start=control_start, seed=seed, eval=True, eval_type='id')
    od_eval_env = KuramotoSivashinskyEnv(Nx=Nx, Lx=Lx, dt=dt, T=T, frameskip=frameskip, max_rl_steps=max_steps,
                                         parametric=parametric, action_scale=action_scale, nr_actuators=nr_actuators,
                                         nr_sensors=nr_sensors, oversampling=oversampling, alpha=alpha, mu=mu,
                                         sigma=sigma, offset=offset, control_start=control_start, seed=seed,  eval=True,
                                         eval_type='od')

    # Load state and action dimensions from environment
    state_dim = env.observation_space.shape[0] + param_dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set seeds
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if log:
        config = {
            "env": env_name,
            "pde": env_name,
            "agent_type": agent_type,
            "max_episode": max_episodes,
            "T": T,
            "Nx": Nx,
            "Lx": Lx,
            "dt": dt,
            "max_rl_steps": max_steps,
            "frameskip": frameskip,
            "oversampling": oversampling,
            "batch_size": batch_size,
            "hidden_dim": h_dim,
            "warmup": warmup,
            "seed": seed,
            "parametric": parametric,
            "evaluation_interval": eval_int,
            "num_actuators": nr_actuators,
            "num_sensors": nr_sensors,
            "action_scale": action_scale,
            "sigma": sigma,
            "alpha": alpha,
            "tau": tau,
            "mu": mu,
            "sparsity_coeff": reg_coeff,
            "trunc_threshold": trunc_threshold,
            "poly_degree": poly_degree,
        }

    # Set seeds
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if agent_type == 'td3':
        agent = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action, h_dim=h_dim, tau=tau,
                    min_action=-max_action, device=device)
    if agent_type == 'td3_noparam':
        agent = TD3_no_param(state_dim=state_dim-1, action_dim=action_dim, max_action=max_action, h_dim=h_dim, tau=tau,
                    min_action=-max_action, device=device)
    if agent_type == 'poly_td3_l0':
        agent = polyTD3_l0(state_dim=state_dim, action_dim=action_dim, max_action=max_action, degree_pi=poly_degree,
                           feature_scale=1.0, h_dim=h_dim, tau=tau, reg_coeff=reg_coeff, min_action=-max_action, device=device)

    replay_buffer = TrainingBuffer(state_dim=state_dim, action_dim=action_dim, device=device)
    train_agent(agent=agent, env=env, eval_env=eval_env, od_eval_env=od_eval_env, max_episodes=max_episodes,
                max_steps=max_steps, warmup=warmup, eval_int=eval_int, max_action=max_action, log=log,
                agent_type=agent_type, save_int=save_int)