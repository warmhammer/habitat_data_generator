import random

import habitat_sim

from hadage.core.settings import make_cfg


def create_simulator(sim_settings):
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    random.seed(sim_settings["seed"])
    sim.seed(sim_settings["seed"])

    return sim


def place_agent(sim, agent_index, start_point, start_rotation=None):
    agent_state = habitat_sim.AgentState()
    agent_state.position = start_point

    if start_rotation is not None:
        agent_state.rotation = start_rotation

    sim.initialize_agent(agent_index, agent_state)

    return sim