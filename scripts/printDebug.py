import logging


def print_episode_summary(env, episode):
    """
    打印episode结束时的智能体和基地状态信息
    """
    separator = f"{'=' * 60}"
    header = f"Episode {episode} 结束 - 剩余单位状态"

    print(f"\n{separator}")
    logging.info(f"\n{separator}")
    print(header)
    logging.info(header)
    print(separator)
    logging.info(separator)

    # 打印红方智能体信息
    red_header = "\n 红方智能体:"
    print(red_header)
    logging.info(red_header)

    red_alive_count = 0
    for agent in env.red_agents:
        if agent.alive:
            red_alive_count += 1
            agent_info = (f"  ID: {agent.id:2d} | 类型: {agent.type.upper():3s} | 血量: {agent.hp:3.0f} | "
                          f"坐标: ({agent.position[0]:6.1f}, {agent.position[1]:6.1f}, {agent.position[2]:6.1f}) | "
                          f"火力: {agent.firepower:3.0f}")
            print(agent_info)
            logging.info(agent_info)

    if red_alive_count == 0:
        no_red_msg = "无存活智能体"
        print(no_red_msg)
        logging.info(no_red_msg)
    else:
        red_summary = f"存活数量: {red_alive_count}/{len(env.red_agents)}"
        print(red_summary)
        logging.info(red_summary)

    # 打印蓝方智能体信息
    blue_header = "\n 蓝方智能体:"
    print(blue_header)
    logging.info(blue_header)

    blue_alive_count = 0
    for agent in env.blue_agents:
        if agent.alive:
            blue_alive_count += 1
            agent_info = (f"  ID: {agent.id:2d} | 类型: {agent.type.upper():3s} | 血量: {agent.hp:3.0f} | "
                          f"坐标: ({agent.position[0]:6.1f}, {agent.position[1]:6.1f}, {agent.position[2]:6.1f}) | "
                          f"火力: {agent.firepower:3.0f}")
            print(agent_info)
            logging.info(agent_info)

    if blue_alive_count == 0:
        no_blue_msg = "无存活智能体"
        print(no_blue_msg)
        logging.info(no_blue_msg)
    else:
        blue_summary = f"存活数量: {blue_alive_count}/{len(env.blue_agents)}"
        print(blue_summary)
        logging.info(blue_summary)

    # 打印基地信息
    base_header = "\n🏭 基地状态:"
    print(base_header)
    logging.info(base_header)

    if hasattr(env, 'base') and env.base:
        base_status = "存活" if env.base.alive else "已摧毁"
        base_info = (f"  血量: {env.base.hp:3.0f} | "
                     f"坐标: ({env.base.position[0]:6.1f}, {env.base.position[1]:6.1f}, {env.base.position[2]:6.1f}) | "
                     f"状态: {base_status}")
        print(base_info)
        logging.info(base_info)
    else:
        no_base_msg = "无基地信息"
        print(no_base_msg)
        logging.info(no_base_msg)

    # 打印总体统计
    total_alive = red_alive_count + blue_alive_count
    total_agents = len(env.red_agents) + len(env.blue_agents)
    stats_header = "\n📊 总体统计:"
    print(stats_header)
    logging.info(stats_header)

    total_msg = f"  总存活智能体: {total_alive}/{total_agents}"
    red_rate_msg = f"  红方存活率: {red_alive_count / len(env.red_agents) * 100:.1f}%"
    blue_rate_msg = f"  蓝方存活率: {blue_alive_count / len(env.blue_agents) * 100:.1f}%"

    print(total_msg)
    logging.info(total_msg)
    print(red_rate_msg)
    logging.info(red_rate_msg)
    print(blue_rate_msg)
    logging.info(blue_rate_msg)

    footer = f"{separator}\n"
    print(footer)
    logging.info(footer)