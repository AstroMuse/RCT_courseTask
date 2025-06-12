import json

def convert_trajectory_format():
    # 读取原始数据
    with open('trajectory_ep108.json', 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # 获取智能体列表和步骤数
    agent_ids = list(source_data.keys())
    agent_ids_sorted = sorted([str(i) for i in range(33)] if all(k.isdigit() for k in agent_ids) else agent_ids)
    
    # 计算总步数（假设所有智能体的步数相同）
    total_steps = len(source_data[agent_ids[0]]) if agent_ids else 0
    
    # 创建目标格式的数据结构
    target_data = {
        "episode": 108,
        "total_steps": total_steps,
        "agent_ids": agent_ids_sorted + ["base"],  # 添加base智能体
        "destroyed_agents": [],  # 需要根据实际情况填充
        "trajectory": []
    }
    
    # 转换轨迹数据
    for step in range(total_steps):
        step_data = {
            "step": step,
            "positions": {}
        }
        
        # 为每个智能体添加该步骤的位置
        for agent_id in agent_ids_sorted:
            if agent_id in source_data and step < len(source_data[agent_id]):
                step_data["positions"][agent_id] = source_data[agent_id][step]
            else:
                # 如果智能体在该步骤不存在，标记为被摧毁
                step_data["positions"][agent_id] = [-1, -1, -1]
        
        # 添加base智能体（假设在原点）
        step_data["positions"]["base"] = [0.0, 0.0, 0.0]
        
        target_data["trajectory"].append(step_data)
    
    # 检测被摧毁的智能体
    destroyed_agents = set()
    for agent_id in agent_ids_sorted:
        if agent_id in source_data:
            for step_pos in source_data[agent_id]:
                if step_pos == [-1, -1, -1]:
                    destroyed_agents.add(agent_id)
                    break
    
    target_data["destroyed_agents"] = list(destroyed_agents)
    
    # 保存转换后的数据
    with open('trajectory_ep1080.json', 'w', encoding='utf-8') as f:
        json.dump(target_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成！")
    print(f"总步数: {total_steps}")
    print(f"智能体数量: {len(agent_ids_sorted)}")
    print(f"被摧毁的智能体: {len(destroyed_agents)}")
    print(f"输出文件: trajectory_ep1080.json")

if __name__ == "__main__":
    convert_trajectory_format()