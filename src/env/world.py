# src/env/world.py
# import numpy as np
# import os
# import json
from typing import Dict, Any
from src.agents.agent import Agent, Base
from src.env.config import ENV_CONFIG
import numpy as np
from src.utils.trajectory_manager import TrajectoryManager


class CombatWorld:
    def __init__(self, output_dir="trajectory_records"):
        # self.global_state_log = []
        self.bounds = ENV_CONFIG["world_bounds"]
        self.max_steps = ENV_CONFIG["max_steps"]
        self.step_count = 0
        self.done = False

        self.red_agents = []
        self.blue_agents = []

        # 轨迹记录
        # self.trajectory = {}
        # self.output_dir = "trajectory_records"
        # os.makedirs(self.output_dir, exist_ok=True)
        self.trajectory_manager = TrajectoryManager(output_dir)
        self._init_agents()

    def _init_agents(self):
        agent_specs = ENV_CONFIG["agent_specs"]

        id_counter = 0
        for _ in range(ENV_CONFIG["red_uav_num"]):
            agent = Agent(id_counter, 'red', 'uav', [0, 0, 1], agent_specs["uav"])
            self.red_agents.append(agent)
            # self.trajectory[agent.id] = [agent.position.tolist()]
            id_counter += 1

        for _ in range(ENV_CONFIG["red_usv_num"]):
            agent = Agent(id_counter, 'red', 'usv', [0, 0, 0], agent_specs["usv"])
            self.red_agents.append(agent)
            # self.trajectory[agent.id] = [agent.position.tolist()]
            id_counter += 1

        for _ in range(ENV_CONFIG["blue_uav_num"]):
            agent = Agent(id_counter, 'blue', 'uav', [200, 200, 1], agent_specs["uav"])
            self.blue_agents.append(agent)
            # self.trajectory[agent.id] = [agent.position.tolist()]
            id_counter += 1

        for _ in range(ENV_CONFIG["blue_usv_num"]):
            agent = Agent(id_counter, 'blue', 'usv', [200, 200, 0], agent_specs["usv"])
            self.blue_agents.append(agent)
            # self.trajectory[agent.id] = [agent.position.tolist()]
            id_counter += 1

        self.base = Base([201, 201, 0], agent_specs["base"]["hp"])

    def get_all_agents(self):
        return self.red_agents + self.blue_agents

    def step(self, red_actions, blue_actions):
        # 修改断言，只检查活着的agent数量
        assert len(red_actions) == sum(agent.alive for agent in self.red_agents)
        assert len(blue_actions) == sum(agent.alive for agent in self.blue_agents)

        self.step_count += 1

        current_obs_dict = self.get_obs_dict()
        # 生成合法目标掩码
        valid_targets_masks = self.get_valid_targets_masks()
        print(f"\n===== Step {self.step_count} =====")

        # 红方动作执行 - 只处理活着的agent
        alive_red_agents = [agent for agent in self.red_agents if agent.alive]
        for agent, (move_action, attack_id) in zip(alive_red_agents, red_actions):
            agent.step(move_action)
            if agent.alive:
                # self.trajectory[agent.id].append(agent.position.tolist())
                agent.attack_target = attack_id

        # 蓝方动作执行 - 只处理活着的agent
        alive_blue_agents = [agent for agent in self.blue_agents if agent.alive]
        for agent, (move_action, attack_id) in zip(alive_blue_agents, blue_actions):
            # 打印agent信息
            # print(f"\nAgent ID: {agent.id}, Team: {agent.team}")
            # print(f"  Observation: {current_obs_dict[agent.id].shape}, Sample: {current_obs_dict[agent.id][0, 0]}")
            # print(f"  Action: Move={move_action}, Attack Target={attack_id}")
            # print(f"  current_obs_dict: {current_obs_dict}")
            agent.step(move_action)
            if agent.alive:
                # self.trajectory[agent.id].append(agent.position.tolist())
                agent.attack_target = attack_id

        # 战斗：根据攻击目标执行攻击
        blue_targets = self.blue_agents + [self.base]
        red_targets = self.red_agents
        # 记录成功攻击的红方agent
        successful_red_attackers = set()
        successful_blue_attackers = set()

        for attacker in self.red_agents:
            if not attacker.alive:
                continue
            target_id = attacker.attack_target
            if target_id == -1 or target_id == len(blue_targets):
                continue
            if attacker.firepower <= 0:
                continue
            if target_id < len(blue_targets):
                target = blue_targets[target_id]
                if target.alive and attacker.in_attack_range(target):
                    target.receive_damage(1)
                    attacker.firepower -= 1  # 消耗1发弹药
                    successful_blue_attackers.add(attacker.id)  # 记录成功攻击的agent
            else:
                print(f"[Warning] Invalid attack_id={target_id} from agent {attacker.id}")

        for attacker in self.blue_agents:
            if not attacker.alive:
                continue
            target_id = attacker.attack_target
            if target_id == -1 or target_id == len(blue_targets):
                continue
            # 检查是否有弹药
            if attacker.firepower <= 0:
                continue
            if target_id < len(red_targets):
                target = red_targets[target_id]
                if target.alive and attacker.in_attack_range(target):
                    target.receive_damage(1)
                    attacker.firepower -= 1  # 消耗1发弹药
                    successful_red_attackers.add(attacker.id)  # 记录成功攻击的agent
            else:
                print(f"[Warning] Invalid attack_id={target_id} from agent {attacker.id}")

        # 默认 team 奖励（胜负、火力、越界等）
        team_red_reward = 0
        team_blue_reward = 0

        if not self.base.alive:
            self.done = True
            team_red_reward += ENV_CONFIG["red_base_reward"]
            team_blue_reward += ENV_CONFIG["blue_base_penalty"]
        elif self.step_count >= self.max_steps:
            self.done = True
            red_firepower = sum([a.firepower for a in self.red_agents if a.alive])
            if red_firepower < self.base.hp:
                team_blue_reward += 2
                team_red_reward -= 2
            if self.base.hp > 0:
                team_blue_reward -= ENV_CONFIG["blue_base_penalty"]
                team_red_reward += ENV_CONFIG["red_base_reward"]

        # 初始化每个 agent 的个体奖励字典
        rewards = {}
        for agent in self.get_all_agents():
            aid = agent.id
            rewards[aid] = 0

            # if agent.alive:
            #     continue
            if not agent.alive:
                continue

            # 越界惩罚
            x, y, z = agent.position[0], agent.position[1], agent.position[2]
            if (
                    x < self.bounds['x'][0] or x > self.bounds['x'][1] or
                    y < self.bounds['y'][0] or y > self.bounds['y'][1] or
                    z < -1 or z > 60
            ):
                rewards[aid] += ENV_CONFIG["out_of_bounds_penalty"]

        for agent_id in successful_red_attackers:
            rewards[agent_id] += 0.5

        for agent_id in successful_blue_attackers:
            rewards[agent_id] += 0.5

        # 新增：红方智能体距离奖励计算
        # base_position = np.array([201, 201, 0])  # 蓝方基地位置
        # spawn_to_base_distance = 284.3  # 出生点到基地的直线距离
        # for agent in self.red_agents:
        #     if not agent.alive:
        #         continue
        #
        #     # 计算智能体到基地的距离
        #     agent_position = np.array(agent.position)
        #     distance_to_base = np.linalg.norm(agent_position - base_position)
        #
        #     # 距离奖励计算：从出生点距离到50的范围内，奖励从0增加到0.05
        #     if distance_to_base <= 30:
        #         # 距离基地50以内，给予最大奖励0.5
        #         distance_reward = 0.3
        #     elif distance_to_base < spawn_to_base_distance:
        #         # 在出生点到基地距离范围内，线性递减奖励
        #         progress = (spawn_to_base_distance - distance_to_base) / (spawn_to_base_distance - 50)
        #         distance_reward = 0.3 * progress
        #     else:
        #         # 距离大于出生点到基地距离，无奖励
        #         distance_reward = 0
        #
        #     rewards[agent.id] += distance_reward

        # 改进的红方智能体距离奖励计算
        base_position = np.array([201, 201, 0])
        spawn_position = np.array([0, 0, 0])  # 红方出生点

        for agent in self.red_agents:
            if not agent.alive:
                continue

            agent_position = np.array(agent.position)
            distance_to_base = np.linalg.norm(agent_position - base_position)

            # 计算智能体是否在朝向基地的正确方向
            spawn_to_base_vector = base_position - spawn_position
            spawn_to_agent_vector = agent_position - spawn_position

            # 计算投影，判断是否越过基地
            projection_length = np.dot(spawn_to_agent_vector, spawn_to_base_vector) / np.linalg.norm(
                spawn_to_base_vector)
            max_projection = np.linalg.norm(spawn_to_base_vector)

            if projection_length > max_projection:
                # 智能体已经越过基地，不给奖励或给予惩罚
                distance_reward = 0  # 越过基地的惩罚
            elif distance_to_base <= 30:
                # 距离基地30米以内，给予最大奖励
                distance_reward = 0.3
            elif projection_length > 0:
                # 在正确方向上前进，根据进度给奖励
                progress = projection_length / max_projection
                distance_reward = 0.3 * progress
            else:
                # 背离基地方向
                distance_reward = 0

            rewards[agent.id] += distance_reward

        # 新增：蓝方距离基地防守奖励
        base_position = np.array([201, 201, 0])  # 蓝方基地位置
        for agent in self.blue_agents:
            if not agent.alive:
                continue

            # 计算智能体到基地的距离
            agent_position = np.array(agent.position)
            distance_to_base = np.linalg.norm(agent_position - base_position)

            # 距离奖励：鼓励蓝方在基地附近防守
            if distance_to_base <= 50:  # 基地50米范围内
                defense_reward = 0.1 * (1 - distance_to_base / 50)  # 距离越近奖励越高

            elif distance_to_base <= 100:  # 基地100米范围内给予较小奖励
                defense_reward = 0.05 * (1 - (distance_to_base - 50) / 50)

            else:
                defense_reward = 0

            rewards[agent.id] += defense_reward

        red_alive_agents = [a for a in self.red_agents if a.alive]
        blue_alive_agents = [a for a in self.blue_agents if a.alive]

        if red_alive_agents:
            per_red_reward = team_red_reward / len(red_alive_agents)
            for agent in red_alive_agents:
                rewards[agent.id] += per_red_reward

        if blue_alive_agents:
            per_blue_reward = team_blue_reward / len(blue_alive_agents)
            for agent in blue_alive_agents:
                rewards[agent.id] += per_blue_reward

        if ENV_CONFIG.get("record_trajectory", True):
            # 修改：使用字典格式而不是列表格式
            agents_data = {
                "red_agents": [],
                "blue_agents": []
            }

            # 添加红方智能体
            for agent in self.red_agents:
                if agent.alive:
                    agents_data["red_agents"].append({
                        "id": agent.id,
                        "position": agent.position.tolist(),
                        "alive": True
                    })
                else:
                    agents_data["red_agents"].append({
                        "id": agent.id,
                        "position": [-1, -1, -1],
                        "alive": False
                    })

            # 添加蓝方智能体
            for agent in self.blue_agents:
                if agent.alive:
                    agents_data["blue_agents"].append({
                        "id": agent.id,
                        "position": agent.position.tolist(),
                        "alive": True
                    })
                else:
                    agents_data["blue_agents"].append({
                        "id": agent.id,
                        "position": [-1, -1, -1],
                        "alive": False
                    })

            # 添加基地信息到蓝方（如果需要）
            if self.base.alive:
                agents_data["blue_agents"].append({
                    "id": "base",
                    "position": self.base.position.tolist(),
                    "alive": True
                })
            else:
                agents_data["blue_agents"].append({
                    "id": "base",
                    "position": [-1, -1, -1],
                    "alive": False
                })

            # 记录当前步骤的智能体位置
            self.trajectory_manager.record_agent_positions(agents_data)

        return self.get_obs_dict(), rewards, self.done, {}

    def get_valid_targets_masks(self):
        """为每个智能体生成合法攻击目标的掩码"""
        masks = {}

        # 红方智能体的掩码
        blue_targets = self.blue_agents + [self.base]
        for agent in self.red_agents:
            if not agent.alive:
                continue

            # 初始化掩码，所有目标都不合法
            mask = [False] * (len(blue_targets) + 1)  # +1 是为了包含"不攻击"选项

            # "不攻击"选项始终合法
            mask[-1] = True

            # 只有在有弹药的情况下才能攻击
            if agent.firepower > 0:
                # 检查每个蓝方目标
                for i, target in enumerate(blue_targets):
                    if target.alive and agent.can_detect(target) and agent.in_attack_range(target):
                        mask[i] = True

            masks[agent.id] = mask

        # 蓝方智能体的掩码
        red_targets = self.red_agents
        for agent in self.blue_agents:
            if not agent.alive:
                continue

            # 初始化掩码，所有目标都不合法
            mask = [False] * (len(red_targets) + 1)  # +1 是为了包含"不攻击"选项

            # "不攻击"选项始终合法
            mask[-1] = True

            # 只有在有弹药的情况下才能攻击
            if agent.firepower > 0:
                # 检查每个红方目标
                for i, target in enumerate(red_targets):
                    if target.alive and agent.can_detect(target) and agent.in_attack_range(target):
                        mask[i] = True

            masks[agent.id] = mask

        return masks

    def _get_global_state(self) -> Dict[str, Any]:
        """获取当前全局状态"""
        return {
            "red_agents": [{
                "id": agent.id,
                "position": agent.position.tolist(),
                "health": agent.hp,
                "alive": agent.alive
            } for agent in self.red_agents],
            "blue_agents": [{
                "id": agent.id,
                "position": agent.position.tolist(),
                "health": agent.hp,
                "alive": agent.alive
            } for agent in self.blue_agents],
            "base_health": self.base.hp,
            "base_alive": self.base.alive
        }

    def get_obs_dict(self):
        obs_dict = {}
        # max_entities = max(len(self.red_agents), len(self.blue_agents)) + 1  # +1 for self
        for agent in self.red_agents + self.blue_agents:
            if not agent.alive:
                continue
            # 创建观测数据结构
            obs = {
                "self": {
                    "position": agent.position.tolist(),
                    "hp": agent.hp,
                    "firepower": agent.firepower,
                    "type": agent.type,  # 'uav' 或 'usv'
                    "id": agent.id
                    # }
                },
                "friends": [],
                "enemies": []
            }

            # 添加友方信息
            friends = self.red_agents if agent.team == 'red' else self.blue_agents
            for friend in friends:
                if friend.id != agent.id:  # 排除自己
                    friend_info = {
                        "position": friend.position.tolist(),
                        "hp": friend.hp,
                        "firepower": friend.firepower,
                        "type": friend.type,
                        "id": friend.id
                    }
                    obs["friends"].append(friend_info)

            # 添加敌方信息
            enemies = self.blue_agents if agent.team == 'red' else self.red_agents
            for enemy in enemies:
                if enemy.alive and agent.can_detect(enemy):
                    enemy_info = {
                        "position": enemy.position.tolist(),
                        "type": enemy.type,
                        "id": enemy.id
                    }
                    obs["enemies"].append(enemy_info)

            # 如果是蓝方，还需要添加基地信息
            if agent.team == 'blue':
                base_info = {
                    "position": self.base.position.tolist(),
                    "hp": self.base.hp,
                    "type": "base",
                    "firepower": 0,
                    "id": -1  # 基地使用特殊ID
                }
                obs["friends"].append(base_info)

            # 如果是红方，检查是否可以探测到蓝方基地
            elif agent.team == 'red' and agent.can_detect(self.base):
                base_info = {
                    "position": self.base.position.tolist(),
                    "type": "base",
                    "id": -1
                }
                obs["enemies"].append(base_info)

            obs_dict[agent.id] = obs

        return obs_dict

    def reset(self):
        self.step_count = 0
        self.done = False
        self.red_agents.clear()
        self.blue_agents.clear()
        # self.trajectory.clear()
        # self.trajectory = {}

        self._init_agents()
        # self.global_state_log = []
        return self.get_obs_dict()

    @property
    def agent_info(self):
        info = {}
        for agent in self.get_all_agents():
            info[agent.id] = {
                "team": agent.team,
                "type": agent.type
            }
        return info
# src/env/world.py
# import numpy as np
# import os
# import json
from typing import Dict, Any
from src.agents.agent import Agent, Base
from src.env.config import ENV_CONFIG
import numpy as np
from src.utils.trajectory_manager import TrajectoryManager


class CombatWorld:
    def __init__(self, output_dir="trajectory_records"):
        # self.global_state_log = []
        self.bounds = ENV_CONFIG["world_bounds"]
        self.max_steps = ENV_CONFIG["max_steps"]
        self.step_count = 0
        self.done = False

        self.red_agents = []
        self.blue_agents = []

        # 轨迹记录
        # self.trajectory = {}
        # self.output_dir = "trajectory_records"
        # os.makedirs(self.output_dir, exist_ok=True)
        self.trajectory_manager = TrajectoryManager(output_dir)
        self._init_agents()

    def _init_agents(self):
        agent_specs = ENV_CONFIG["agent_specs"]

        id_counter = 0
        for _ in range(ENV_CONFIG["red_uav_num"]):
            agent = Agent(id_counter, 'red', 'uav', [0, 0, 1], agent_specs["uav"])
            self.red_agents.append(agent)
            # self.trajectory[agent.id] = [agent.position.tolist()]
            id_counter += 1

        for _ in range(ENV_CONFIG["red_usv_num"]):
            agent = Agent(id_counter, 'red', 'usv', [0, 0, 0], agent_specs["usv"])
            self.red_agents.append(agent)
            # self.trajectory[agent.id] = [agent.position.tolist()]
            id_counter += 1

        for _ in range(ENV_CONFIG["blue_uav_num"]):
            agent = Agent(id_counter, 'blue', 'uav', [200, 200, 1], agent_specs["uav"])
            self.blue_agents.append(agent)
            # self.trajectory[agent.id] = [agent.position.tolist()]
            id_counter += 1

        for _ in range(ENV_CONFIG["blue_usv_num"]):
            agent = Agent(id_counter, 'blue', 'usv', [200, 200, 0], agent_specs["usv"])
            self.blue_agents.append(agent)
            # self.trajectory[agent.id] = [agent.position.tolist()]
            id_counter += 1

        self.base = Base([201, 201, 0], agent_specs["base"]["hp"])

    def get_all_agents(self):
        return self.red_agents + self.blue_agents

    def step(self, red_actions, blue_actions):
        # 修改断言，只检查活着的agent数量
        assert len(red_actions) == sum(agent.alive for agent in self.red_agents)
        assert len(blue_actions) == sum(agent.alive for agent in self.blue_agents)

        self.step_count += 1

        current_obs_dict = self.get_obs_dict()
        # 生成合法目标掩码
        valid_targets_masks = self.get_valid_targets_masks()
        print(f"\n===== Step {self.step_count} =====")

        # 红方动作执行 - 只处理活着的agent
        alive_red_agents = [agent for agent in self.red_agents if agent.alive]
        for agent, (move_action, attack_id) in zip(alive_red_agents, red_actions):
            agent.step(move_action)
            if agent.alive:
                # self.trajectory[agent.id].append(agent.position.tolist())
                agent.attack_target = attack_id

        # 蓝方动作执行 - 只处理活着的agent
        alive_blue_agents = [agent for agent in self.blue_agents if agent.alive]
        for agent, (move_action, attack_id) in zip(alive_blue_agents, blue_actions):
            # 打印agent信息
            # print(f"\nAgent ID: {agent.id}, Team: {agent.team}")
            # print(f"  Observation: {current_obs_dict[agent.id].shape}, Sample: {current_obs_dict[agent.id][0, 0]}")
            # print(f"  Action: Move={move_action}, Attack Target={attack_id}")
            # print(f"  current_obs_dict: {current_obs_dict}")
            agent.step(move_action)
            if agent.alive:
                # self.trajectory[agent.id].append(agent.position.tolist())
                agent.attack_target = attack_id

        # 战斗：根据攻击目标执行攻击
        blue_targets = self.blue_agents + [self.base]
        red_targets = self.red_agents
        # 记录成功攻击的红方agent
        successful_red_attackers = set()
        successful_blue_attackers = set()

        for attacker in self.red_agents:
            if not attacker.alive:
                continue
            target_id = attacker.attack_target
            if target_id == -1 or target_id == len(blue_targets):
                continue
            if attacker.firepower <= 0:
                continue
            if target_id < len(blue_targets):
                target = blue_targets[target_id]
                if target.alive and attacker.in_attack_range(target):
                    target.receive_damage(1)
                    attacker.firepower -= 1  # 消耗1发弹药
                    successful_blue_attackers.add(attacker.id)  # 记录成功攻击的agent
            else:
                print(f"[Warning] Invalid attack_id={target_id} from agent {attacker.id}")

        for attacker in self.blue_agents:
            if not attacker.alive:
                continue
            target_id = attacker.attack_target
            if target_id == -1 or target_id == len(blue_targets):
                continue
            # 检查是否有弹药
            if attacker.firepower <= 0:
                continue
            if target_id < len(red_targets):
                target = red_targets[target_id]
                if target.alive and attacker.in_attack_range(target):
                    target.receive_damage(1)
                    attacker.firepower -= 1  # 消耗1发弹药
                    successful_red_attackers.add(attacker.id)  # 记录成功攻击的agent
            else:
                print(f"[Warning] Invalid attack_id={target_id} from agent {attacker.id}")

        # 默认 team 奖励（胜负、火力、越界等）
        team_red_reward = 0
        team_blue_reward = 0

        if not self.base.alive:
            self.done = True
            team_red_reward += ENV_CONFIG["red_base_reward"]
            team_blue_reward += ENV_CONFIG["blue_base_penalty"]
        elif self.step_count >= self.max_steps:
            self.done = True
            red_firepower = sum([a.firepower for a in self.red_agents if a.alive])
            if red_firepower < self.base.hp:
                team_blue_reward += 2
                team_red_reward -= 2
            if self.base.hp > 0:
                team_blue_reward -= ENV_CONFIG["blue_base_penalty"]
                team_red_reward += ENV_CONFIG["red_base_reward"]

        # 初始化每个 agent 的个体奖励字典
        rewards = {}
        for agent in self.get_all_agents():
            aid = agent.id
            rewards[aid] = 0

            # if agent.alive:
            #     continue
            if not agent.alive:
                continue

            # 越界惩罚
            x, y, z = agent.position[0], agent.position[1], agent.position[2]
            if (
                    x < self.bounds['x'][0] or x > self.bounds['x'][1] or
                    y < self.bounds['y'][0] or y > self.bounds['y'][1] or
                    z < -1 or z > 60
            ):
                rewards[aid] += ENV_CONFIG["out_of_bounds_penalty"]

        for agent_id in successful_red_attackers:
            rewards[agent_id] += 0.5

        for agent_id in successful_blue_attackers:
            rewards[agent_id] += 0.5

        # 新增：红方智能体距离奖励计算
        # base_position = np.array([201, 201, 0])  # 蓝方基地位置
        # spawn_to_base_distance = 284.3  # 出生点到基地的直线距离
        # for agent in self.red_agents:
        #     if not agent.alive:
        #         continue
        #
        #     # 计算智能体到基地的距离
        #     agent_position = np.array(agent.position)
        #     distance_to_base = np.linalg.norm(agent_position - base_position)
        #
        #     # 距离奖励计算：从出生点距离到50的范围内，奖励从0增加到0.05
        #     if distance_to_base <= 30:
        #         # 距离基地50以内，给予最大奖励0.5
        #         distance_reward = 0.3
        #     elif distance_to_base < spawn_to_base_distance:
        #         # 在出生点到基地距离范围内，线性递减奖励
        #         progress = (spawn_to_base_distance - distance_to_base) / (spawn_to_base_distance - 50)
        #         distance_reward = 0.3 * progress
        #     else:
        #         # 距离大于出生点到基地距离，无奖励
        #         distance_reward = 0
        #
        #     rewards[agent.id] += distance_reward

        # 改进的红方智能体距离奖励计算
        base_position = np.array([201, 201, 0])
        spawn_position = np.array([0, 0, 0])  # 红方出生点

        for agent in self.red_agents:
            if not agent.alive:
                continue

            agent_position = np.array(agent.position)
            distance_to_base = np.linalg.norm(agent_position - base_position)

            # 计算智能体是否在朝向基地的正确方向
            spawn_to_base_vector = base_position - spawn_position
            spawn_to_agent_vector = agent_position - spawn_position

            # 计算投影，判断是否越过基地
            projection_length = np.dot(spawn_to_agent_vector, spawn_to_base_vector) / np.linalg.norm(
                spawn_to_base_vector)
            max_projection = np.linalg.norm(spawn_to_base_vector)

            if projection_length > max_projection:
                # 智能体已经越过基地，不给奖励或给予惩罚
                distance_reward = 0  # 越过基地的惩罚
            elif distance_to_base <= 30:
                # 距离基地30米以内，给予最大奖励
                distance_reward = 0.3
            elif projection_length > 0:
                # 在正确方向上前进，根据进度给奖励
                progress = projection_length / max_projection
                distance_reward = 0.3 * progress
            else:
                # 背离基地方向
                distance_reward = 0

            rewards[agent.id] += distance_reward

        # 新增：蓝方距离基地防守奖励
        base_position = np.array([201, 201, 0])  # 蓝方基地位置
        for agent in self.blue_agents:
            if not agent.alive:
                continue

            # 计算智能体到基地的距离
            agent_position = np.array(agent.position)
            distance_to_base = np.linalg.norm(agent_position - base_position)

            # 距离奖励：鼓励蓝方在基地附近防守
            if distance_to_base <= 50:  # 基地50米范围内
                defense_reward = 0.1 * (1 - distance_to_base / 50)  # 距离越近奖励越高

            elif distance_to_base <= 100:  # 基地100米范围内给予较小奖励
                defense_reward = 0.05 * (1 - (distance_to_base - 50) / 50)

            else:
                defense_reward = 0

            rewards[agent.id] += defense_reward

        red_alive_agents = [a for a in self.red_agents if a.alive]
        blue_alive_agents = [a for a in self.blue_agents if a.alive]

        if red_alive_agents:
            per_red_reward = team_red_reward / len(red_alive_agents)
            for agent in red_alive_agents:
                rewards[agent.id] += per_red_reward

        if blue_alive_agents:
            per_blue_reward = team_blue_reward / len(blue_alive_agents)
            for agent in blue_alive_agents:
                rewards[agent.id] += per_blue_reward

        if ENV_CONFIG.get("record_trajectory", True):
            # 修改：使用字典格式而不是列表格式
            agents_data = {
                "red_agents": [],
                "blue_agents": []
            }

            # 添加红方智能体
            for agent in self.red_agents:
                if agent.alive:
                    agents_data["red_agents"].append({
                        "id": agent.id,
                        "position": agent.position.tolist(),
                        "alive": True
                    })
                else:
                    agents_data["red_agents"].append({
                        "id": agent.id,
                        "position": [-1, -1, -1],
                        "alive": False
                    })

            # 添加蓝方智能体
            for agent in self.blue_agents:
                if agent.alive:
                    agents_data["blue_agents"].append({
                        "id": agent.id,
                        "position": agent.position.tolist(),
                        "alive": True
                    })
                else:
                    agents_data["blue_agents"].append({
                        "id": agent.id,
                        "position": [-1, -1, -1],
                        "alive": False
                    })

            # 添加基地信息到蓝方（如果需要）
            if self.base.alive:
                agents_data["blue_agents"].append({
                    "id": "base",
                    "position": self.base.position.tolist(),
                    "alive": True
                })
            else:
                agents_data["blue_agents"].append({
                    "id": "base",
                    "position": [-1, -1, -1],
                    "alive": False
                })

            # 记录当前步骤的智能体位置
            self.trajectory_manager.record_agent_positions(agents_data)

        return self.get_obs_dict(), rewards, self.done, {}

    def get_valid_targets_masks(self):
        """为每个智能体生成合法攻击目标的掩码"""
        masks = {}

        # 红方智能体的掩码
        blue_targets = self.blue_agents + [self.base]
        for agent in self.red_agents:
            if not agent.alive:
                continue

            # 初始化掩码，所有目标都不合法
            mask = [False] * (len(blue_targets) + 1)  # +1 是为了包含"不攻击"选项

            # "不攻击"选项始终合法
            mask[-1] = True

            # 只有在有弹药的情况下才能攻击
            if agent.firepower > 0:
                # 检查每个蓝方目标
                for i, target in enumerate(blue_targets):
                    if target.alive and agent.can_detect(target) and agent.in_attack_range(target):
                        mask[i] = True

            masks[agent.id] = mask

        # 蓝方智能体的掩码
        red_targets = self.red_agents
        for agent in self.blue_agents:
            if not agent.alive:
                continue

            # 初始化掩码，所有目标都不合法
            mask = [False] * (len(red_targets) + 1)  # +1 是为了包含"不攻击"选项

            # "不攻击"选项始终合法
            mask[-1] = True

            # 只有在有弹药的情况下才能攻击
            if agent.firepower > 0:
                # 检查每个红方目标
                for i, target in enumerate(red_targets):
                    if target.alive and agent.can_detect(target) and agent.in_attack_range(target):
                        mask[i] = True

            masks[agent.id] = mask

        return masks

    def _get_global_state(self) -> Dict[str, Any]:
        """获取当前全局状态"""
        return {
            "red_agents": [{
                "id": agent.id,
                "position": agent.position.tolist(),
                "health": agent.hp,
                "alive": agent.alive
            } for agent in self.red_agents],
            "blue_agents": [{
                "id": agent.id,
                "position": agent.position.tolist(),
                "health": agent.hp,
                "alive": agent.alive
            } for agent in self.blue_agents],
            "base_health": self.base.hp,
            "base_alive": self.base.alive
        }

    def get_obs_dict(self):
        obs_dict = {}
        # max_entities = max(len(self.red_agents), len(self.blue_agents)) + 1  # +1 for self
        for agent in self.red_agents + self.blue_agents:
            if not agent.alive:
                continue
            # 创建观测数据结构
            obs = {
                "self": {
                    "position": agent.position.tolist(),
                    "hp": agent.hp,
                    "firepower": agent.firepower,
                    "type": agent.type,  # 'uav' 或 'usv'
                    "id": agent.id
                    # }
                },
                "friends": [],
                "enemies": []
            }

            # 添加友方信息
            friends = self.red_agents if agent.team == 'red' else self.blue_agents
            for friend in friends:
                if friend.id != agent.id:  # 排除自己
                    friend_info = {
                        "position": friend.position.tolist(),
                        "hp": friend.hp,
                        "firepower": friend.firepower,
                        "type": friend.type,
                        "id": friend.id
                    }
                    obs["friends"].append(friend_info)

            # 添加敌方信息
            enemies = self.blue_agents if agent.team == 'red' else self.red_agents
            for enemy in enemies:
                if enemy.alive and agent.can_detect(enemy):
                    enemy_info = {
                        "position": enemy.position.tolist(),
                        "type": enemy.type,
                        "id": enemy.id
                    }
                    obs["enemies"].append(enemy_info)

            # 如果是蓝方，还需要添加基地信息
            if agent.team == 'blue':
                base_info = {
                    "position": self.base.position.tolist(),
                    "hp": self.base.hp,
                    "type": "base",
                    "firepower": 0,
                    "id": -1  # 基地使用特殊ID
                }
                obs["friends"].append(base_info)

            # 如果是红方，检查是否可以探测到蓝方基地
            elif agent.team == 'red' and agent.can_detect(self.base):
                base_info = {
                    "position": self.base.position.tolist(),
                    "type": "base",
                    "id": -1
                }
                obs["enemies"].append(base_info)

            obs_dict[agent.id] = obs

        return obs_dict

    def reset(self):
        self.step_count = 0
        self.done = False
        self.red_agents.clear()
        self.blue_agents.clear()
        # self.trajectory.clear()
        # self.trajectory = {}

        self._init_agents()
        # self.global_state_log = []
        return self.get_obs_dict()

    @property
    def agent_info(self):
        info = {}
        for agent in self.get_all_agents():
            info[agent.id] = {
                "team": agent.team,
                "type": agent.type
            }
        return info