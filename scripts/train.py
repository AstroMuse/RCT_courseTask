import torch
from src.env.world import CombatWorld
from src.ppo.model import PolicyManager
from src.ppo.ppo import PPOAgent
from src.env.config import ENV_CONFIG
import numpy as np
import os
import logging
from datetime import datetime
from src.utils.plot_kl_rew import plot_layered_training_curves, plot_kl_divergence_curve
from printDebug import print_episode_summary

# from src.ppo.obsDataProcess import ObsEncoder
# 在脚本顶部加入（如未已有）
save_dir = "weights"
os.makedirs(save_dir, exist_ok=True)
os.makedirs("log", exist_ok=True)

# 1. 设置日志文件名（按时间命名，避免覆盖）
log_filename = f"log/train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# 2. 配置 logging
logging.basicConfig(
    level=logging.DEBUG,  # 记录 INFO 及以上级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),  # 输出到文件
        # logging.StreamHandler()  # 输出到控制台（可选）
    ]
)

# GPU优化设置
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 优化CUDNN性能
    torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提升性能
    print(f"CUDA可用，GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
else:
    print("CUDA不可用，使用CPU")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_DIM = 128  # 暂定，根据观察设计修改
ACT_DIM = 3  # x, y, z 速度控制维度

# 初始化环境与策略管理器
env = CombatWorld()
# output_dir = "experiments/exp1/trajectories"
# env = CombatWorld(output_dir=output_dir)
# policy_manager = PolicyManager(obs_dim=OBS_DIM, act_dim=ACT_DIM, device=DEVICE)
# num_targets = ENV_CONFIG["blue_uav_num"] + ENV_CONFIG["blue_usv_num"] + 1
# policy_manager = PolicyManager(OBS_DIM, ACT_DIM, num_targets, DEVICE)
red_targets_num = ENV_CONFIG["blue_uav_num"] + ENV_CONFIG["blue_usv_num"] + 1  # 蓝方智能体 + 基地
blue_targets_num = ENV_CONFIG["red_uav_num"] + ENV_CONFIG["red_usv_num"]  # 红方智能体

# 修改PolicyManager的初始化方式
policy_manager = PolicyManager(OBS_DIM, ACT_DIM, red_targets_num, blue_targets_num, DEVICE)

# 为每类 agent 创建 PPO 优化器
agents = {
    'red_uav': PPOAgent(policy_manager.red_uav),
    'red_usv': PPOAgent(policy_manager.red_usv),
    'blue_uav': PPOAgent(policy_manager.blue_uav),
    'blue_usv': PPOAgent(policy_manager.blue_usv)
}

# 添加用于绘制曲线的列表分层奖励记录结构
reward_history = {
    'red_team': [],  # 红方团队总奖励
    'blue_team': [],  # 蓝方团队总奖励
    'red_uav': [],  # 红方UAV平均奖励
    'red_usv': [],  # 红方USV平均奖励
    'blue_uav': [],  # 蓝方UAV平均奖励
    'blue_usv': [],  # 蓝方USV平均奖励
    'total': []  # 全局总奖励
}
# 胜负统计
win_statistics = {
    'red_wins': 0,
    'blue_wins': 0,
    'draws': 0,
    'red_win_rate': [],
    'blue_win_rate': []
}
kl_divergence_history = []
episode_rewards = []

# 训练参数
EPISODES = 200
STEPS_PER_EPISODE = 200
BATCH_SIZE = 1024
UPDATE_EPOCHS = 3

logging.info("开始训练...")
for episode in range(EPISODES):
    logging.info(f"[Episode {episode}] 训练中...")
    print(f"[Episode {episode}] 训练中...")

    obs_dict = env.reset()
    # 添加：开始新episode的轨迹记录
    env.trajectory_manager.start_new_episode(episode)
    # 记录初始位置
    if ENV_CONFIG.get("record_trajectory", True):
        agents_data = {
            "red_agents": [],
            "blue_agents": []
        }

        # 添加红方智能体初始位置
        for agent in env.red_agents:
            agents_data["red_agents"].append({
                "id": agent.id,
                "position": agent.position.tolist(),
                "alive": True
            })

        # 添加蓝方智能体初始位置
        for agent in env.blue_agents:
            agents_data["blue_agents"].append({
                "id": agent.id,
                "position": agent.position.tolist(),
                "alive": True
            })

        # 添加基地初始位置
        agents_data["blue_agents"].append({
            "id": "base",
            "position": env.base.position.tolist(),
            "alive": True
        })

        # 记录初始位置
        env.trajectory_manager.record_agent_positions(agents_data)
    buffer = {k: {'obs': [], 'acts': [], 'logps': [], 'rews': [], 'dones': [], 'vals': [], 'encoded_obs': []} for k in
              agents.keys()}
    # 新的分层奖励统计
    team_rewards = {'red': 0, 'blue': 0}
    type_rewards = {'red_uav': [], 'red_usv': [], 'blue_uav': [], 'blue_usv': []}
    episode_reward = 0
    done = False  # 初始化done变量

    for step in range(STEPS_PER_EPISODE):
        # logging.info(f"[STEP: {step}] env更新中...")
        print(f"[STEP: {step}] env更新中...")
        actions = {}
        logps = {}
        # 获取观测和合法目标掩码
        obs_dict = env.get_obs_dict()  # noqa
        valid_targets_masks = env.get_valid_targets_masks()

        for aid, obs in obs_dict.items():
            # logging.info(f"[Agent {aid}], [obs: {obs}]")  # 检查观测数据是否包含 'firepower'
            team, agent_type = env.agent_info[aid]['team'], env.agent_info[aid]['type']
            policy = policy_manager.get_policy(team, agent_type)

            # 获取该智能体的合法目标掩码
            valid_mask = valid_targets_masks.get(aid, None)
            (move_action, attack_target), logp, _ = policy.act(obs)
            if isinstance(move_action, np.ndarray) and move_action.shape == (1, 3):
                move_action = move_action.squeeze(0)
            actions[aid] = (move_action, attack_target)
            logps[aid] = logp

            key = f"{team}_{agent_type}"
            buffer[key]['obs'].append(obs)
            buffer[key]['acts'].append((move_action, attack_target))
            buffer[key]['logps'].append(logp)

        red_actions = []
        blue_actions = []
        for aid, act in actions.items():
            team = env.agent_info[aid]['team']
            if team == 'red':
                red_actions.append(act)
            elif team == 'blue':
                blue_actions.append(act)

        obs_next, rewards, done, _ = env.step(red_actions, blue_actions)

        for aid, rew in rewards.items():
            episode_reward += rew
            if aid not in obs_dict:  # 跳过已销毁的智能体
                continue
            team, agent_type = env.agent_info[aid]['team'], env.agent_info[aid]['type']
            team_rewards[team] += rew  # 团队奖励统计

            key = f"{team}_{agent_type}"
            type_rewards[key].append(rew)
            # 日志记录
            logging.info(f"[Episode {episode}] Agent {aid} ({team}_{agent_type}): Reward={rew:.3f}")
            buffer[key]['rews'].append(rew)
            buffer[key]['dones'].append(done)
            policy = policy_manager.get_policy(team, agent_type)
            # val = policy.value_net(torch.tensor(obs_dict[aid], dtype=torch.float32).to(DEVICE)).item()
            # logging.info(f"[Current aid: {aid}]")
            val = policy.value_net(obs_dict[aid]).item()
            buffer[key]['vals'].append(val)

        obs_dict = obs_next
        if done:
            break

    # 在episode结束后添加：保存轨迹
    env.trajectory_manager.save_episode_trajectory()

    # 胜负判定和统计
    red_alive = len([a for a in env.red_agents if a.alive])
    blue_alive = len([a for a in env.blue_agents if a.alive])
    base_alive = env.base.alive

    if not base_alive:  # 红方摧毁基地获胜
        win_statistics['red_wins'] += 1
        winner = 'Red'
    elif red_alive == 0:  # 红方全灭，蓝方获胜
        win_statistics['blue_wins'] += 1
        winner = 'Blue'
    elif done and base_alive:  # 超时且基地存活，蓝方获胜
        win_statistics['blue_wins'] += 1
        winner = 'Blue'
    else:
        win_statistics['draws'] += 1
        winner = 'Draw'

    # 计算胜率
    total_games = episode + 1
    red_win_rate = win_statistics['red_wins'] / total_games
    blue_win_rate = win_statistics['blue_wins'] / total_games
    win_statistics['red_win_rate'].append(red_win_rate)
    win_statistics['blue_win_rate'].append(blue_win_rate)

    # 最后状态的 value 估计
    for key in buffer:
        # logging.info(f"[obs_dict[list(obs_dict.keys())[0]]: {obs_dict[list(obs_dict.keys())[0]]}]")
        # last_obs = torch.tensor(obs_dict[list(obs_dict.keys())[0]], dtype=torch.float32).to(DEVICE)
        # next_value = policy_manager.get_policy(*key.split('_')).value_net(last_obs).item()
        aid = list(obs_dict.keys())[0]
        last_obs = obs_dict[aid]  # 保持原始字典结构
        next_value = policy_manager.get_policy(*key.split('_')).value_net(last_obs).item()
        advs, rets = agents[key].compute_gae(
            buffer[key]['rews'], buffer[key]['vals'], buffer[key]['dones'], next_value)
        buffer[key]['advs'] = advs
        buffer[key]['rets'] = rets

    # 日志记录团队统计
    logging.info(f"[Episode {episode}] Team Rewards - Red: {team_rewards['red']:.2f}, Blue: {team_rewards['blue']:.2f}")
    logging.info(
        f"[Episode {episode}] Winner: {winner}, Red WinRate: {red_win_rate:.3f}, Blue WinRate: {blue_win_rate:.3f}")
    episode_rewards.append(episode_reward)

    # 添加：更新reward_history
    reward_history['red_team'].append(team_rewards['red'])
    reward_history['blue_team'].append(team_rewards['blue'])
    reward_history['total'].append(episode_reward)

    # 计算并添加各类型智能体的平均奖励
    reward_history['red_uav'].append(np.mean(type_rewards['red_uav']) if type_rewards['red_uav'] else 0)
    reward_history['red_usv'].append(np.mean(type_rewards['red_usv']) if type_rewards['red_usv'] else 0)
    reward_history['blue_uav'].append(np.mean(type_rewards['blue_uav']) if type_rewards['blue_uav'] else 0)
    reward_history['blue_usv'].append(np.mean(type_rewards['blue_usv']) if type_rewards['blue_usv'] else 0)

    # 执行 PPO 更新并计算KL散度
    total_kl_div = 0
    print_episode_summary(env, episode)
    for key in agents:
        # 在update方法中添加KL散度计算
        print(f"now in key[{key}]...")
        kl_div = agents[key].update(
            buffer[key]['obs'],
            buffer[key]['acts'],
            buffer[key]['logps'],
            buffer[key]['rets'],
            buffer[key]['advs'],
            epochs=UPDATE_EPOCHS,
            batch_size=BATCH_SIZE
        )
        total_kl_div += kl_div if kl_div is not None else 0

        # 记录KL散度
    kl_divergence_history.append(total_kl_div / len(agents))
    print(f"[Episode {episode}] Finished, Reward: {episode_reward:.2f}, KL Div: {total_kl_div / len(agents):.6f}")

    if episode % 10 == 0:
        # 为每个agent类型分别保存权重
        for agent_key in agents.keys():
            policy_path = os.path.join(save_dir, f"{agent_key}_policy_net_ep{episode}.pt")
            value_path = os.path.join(save_dir, f"{agent_key}_value_net_ep{episode}.pt")
            torch.save(agents[agent_key].policy.policy_net.state_dict(), policy_path)
            torch.save(agents[agent_key].policy.value_net.state_dict(), value_path)
            print(f"[Episode {episode}] {agent_key} weights saved.")

    # 绘制和保存曲线
    # plot_training_curves(episode_rewards, kl_divergence_history, episode)
    plot_layered_training_curves(reward_history, win_statistics, kl_divergence_history, episode)
    plot_kl_divergence_curve(kl_divergence_history, episode)

# 训练结束后绘制最终曲线
plot_layered_training_curves(reward_history, win_statistics, kl_divergence_history, EPISODES - 1, final=True)
