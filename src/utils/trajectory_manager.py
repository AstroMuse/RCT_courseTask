import json
import os
# from datetime import datetime
from typing import Dict, Any


class TrajectoryManager:
    def __init__(self, output_dir: str = "trajectory_records"):
        self.output_dir = output_dir
        self.episode_count = 0
        self.current_episode_step = 0
        self.agent_positions = {}  # {step: {agent_id: [x, y, z]}}
        self.destroyed_agents = set()  # 记录已被摧毁的智能体ID
        self.all_agent_ids = set()  # 记录所有出现过的智能体ID

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    def start_new_episode(self, episode_num: int):
        """开始新的episode记录"""
        self.episode_count = episode_num
        self.current_episode_step = 0
        self.agent_positions = {}
        self.destroyed_agents = set()
        self.all_agent_ids = set()

    def record_agent_positions(self, agents_data: Dict[str, Any]):
        step_positions = {}

        # 处理不同的输入格式
        if 'red_agents' in agents_data and 'blue_agents' in agents_data:
            # 处理智能体列表格式
            all_agents = agents_data.get('red_agents', []) + agents_data.get('blue_agents', [])
            for agent in all_agents:
                agent_id = agent['id']
                self.all_agent_ids.add(agent_id)

                if agent.get('alive', True):
                    # 智能体存活，记录真实坐标
                    position = agent['position']
                    if isinstance(position, list) and len(position) >= 3:
                        step_positions[agent_id] = position[:3]
                    else:
                        step_positions[agent_id] = [-1, -1, -1]
                else:
                    # 智能体被摧毁
                    self.destroyed_agents.add(agent_id)
                    step_positions[agent_id] = [-1, -1, -1]

        elif 'observations' in agents_data:
            # 处理观测字典格式
            obs_dict = agents_data['observations']
            for agent_id, obs in obs_dict.items():
                self.all_agent_ids.add(agent_id)
                if 'self' in obs and 'position' in obs['self']:
                    position = obs['self']['position']
                    step_positions[agent_id] = position[:3] if len(position) >= 3 else [-1, -1, -1]
                else:
                    step_positions[agent_id] = [-1, -1, -1]

        else:
            # 直接处理智能体ID和位置的字典
            for agent_id, data in agents_data.items():
                self.all_agent_ids.add(agent_id)
                if isinstance(data, dict) and 'position' in data:
                    if data.get('alive', True):
                        position = data['position']
                        step_positions[agent_id] = position[:3] if len(position) >= 3 else [-1, -1, -1]
                    else:
                        self.destroyed_agents.add(agent_id)
                        step_positions[agent_id] = [-1, -1, -1]
                elif isinstance(data, list) and len(data) >= 3:
                    # 直接传入坐标列表
                    step_positions[agent_id] = data[:3]
                else:
                    step_positions[agent_id] = [-1, -1, -1]

        # 为所有已知的智能体补充坐标记录
        for agent_id in self.all_agent_ids:
            if agent_id not in step_positions:
                if agent_id in self.destroyed_agents:
                    # 已被摧毁的智能体记录为(-1,-1,-1)
                    step_positions[agent_id] = [-1, -1, -1]
                else:
                    # 未在当前步骤出现的智能体也记录为(-1,-1,-1)
                    step_positions[agent_id] = [-1, -1, -1]

        # 记录当前步骤的位置数据
        self.agent_positions[self.current_episode_step] = step_positions
        self.current_episode_step += 1

    def record_step(self, step_data: Dict[str, Any]):
        """兼容原有接口的记录方法"""
        # 从step_data中提取智能体位置信息
        if 'observations' in step_data:
            self.record_agent_positions({'observations': step_data['observations']})
        elif 'agents' in step_data:
            self.record_agent_positions(step_data['agents'])
        else:
            # 尝试直接解析step_data
            self.record_agent_positions(step_data)

    def record_global_state(self, global_state: Dict[str, Any]):
        """记录全局状态（保持兼容性，但在简化版本中不使用）"""
        pass

    def save_episode_trajectory(self):
        """保存当前episode的轨迹数据"""
        if not self.agent_positions:
            print(f"Episode {self.episode_count}: 没有轨迹数据可保存")
            return

        # 构建轨迹数据结构
        trajectory_data = {
            "episode": self.episode_count,
            "total_steps": len(self.agent_positions),
            "agent_ids": sorted([str(agent_id) for agent_id in self.all_agent_ids]),
            "destroyed_agents": sorted([str(agent_id) for agent_id in self.destroyed_agents]),
            "trajectory": []
        }

        # 按步骤顺序构建轨迹
        for step in sorted(self.agent_positions.keys()):
            step_data = {
                "step": step,
                "positions": {}
            }

            # 为每个智能体记录位置
            for agent_id in sorted([str(agent_id) for agent_id in self.all_agent_ids]):
                if agent_id in self.agent_positions[step]:
                    step_data["positions"][str(agent_id)] = self.agent_positions[step][agent_id]
                else:
                    # 如果某个智能体在这一步没有记录，补充为(-1,-1,-1)
                    step_data["positions"][str(agent_id)] = [-1, -1, -1]

            trajectory_data["trajectory"].append(step_data)

        # 保存JSON文件
        trajectory_file = os.path.join(
            self.output_dir,
            f"trajectory_ep{self.episode_count:03d}.json"
        )

        try:
            with open(trajectory_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

            print(
                f"轨迹已保存: Episode {self.episode_count}, 总步数: {len(self.agent_positions)}, 智能体数量: {len(self.all_agent_ids)}")
            print(f"文件路径: {trajectory_file}")

        except Exception as e:
            print(f"保存轨迹文件时出错: {e}")

    def get_episode_summary(self) -> Dict[str, Any]:
        """获取当前episode的摘要信息"""
        return {
            "episode": self.episode_count,
            "total_steps": len(self.agent_positions),
            "current_step": self.current_episode_step,
            "total_agents": len(self.all_agent_ids),
            "destroyed_agents": len(self.destroyed_agents)
        }

    def mark_agent_destroyed(self, agent_id: int):
        """手动标记智能体为已摧毁状态"""
        self.destroyed_agents.add(agent_id)
        self.all_agent_ids.add(agent_id)