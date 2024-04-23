import random
from concurrent.futures import ThreadPoolExecutor

class State:
    def __init__(self, battery_soc, predicted_prices, hour_of_day):
        self.battery_soc = battery_soc
        self.predicted_prices = predicted_prices
        self.hour_of_day = hour_of_day
        self.pv_power 

    def is_terminal(self, step, max_steps):
        return step >= max_steps

class Action:
    def __init__(self, solar_to_battery, charge_from_grid):
        self.solar_to_battery = solar_to_battery
        self.charge_from_grid = charge_from_grid

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.total_reward = 0
        self.visits = 0
        self.untried_actions = self.get_possible_actions()

    def get_possible_actions(self, state):
        actions = []
        # Generating action space
        for solar_to_battery in [i * 0.1 for i in range(state.pv_power)]:
            for charge_from_grid in [j * 0.1 for j in range(-10, 11)]:
                actions.append(Action(solar_to_battery, charge_from_grid))
        return actions

    def select_child(self):
        # Use UCB1 formula for child selection
        import math
        log_parent_visits = math.log(self.visits)
        return max(self.children, key=lambda x: x.total_reward / x.visits + math.sqrt(2 * log_parent_visits / x.visits))

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward

def transition(state, action):
    # Modify state based on action
    new_battery_soc = state.battery_soc + (action.solar_to_battery - abs(action.charge_from_grid)) * 5 / 13
    new_battery_soc = max(0, min(new_battery_soc, 100))
    new_predicted_prices = state.predicted_prices[1:]  # Shift prices
    new_hour_of_day = (state.hour_of_day + 1) % 24
    return State(new_battery_soc, new_predicted_prices, new_hour_of_day)

def get_reward(action, state):
    # Adapted reward function based on provided specs
    solar_to_battery, charge_from_grid = action.solar_to_battery, action.charge_from_grid
    solar_to_grid = 1 - solar_to_battery
    battery_to_grid = -min(charge_from_grid, 0)
    grid_to_battery = max(charge_from_grid, 0)

    potential_battery_soc = state.battery_soc + (solar_to_battery + grid_to_battery) / 13
    potential_battery_soc = max(0, min(potential_battery_soc, 1))

    battery_soc_min = 0.2
    battery_soc_max = 0.8

    overdischarge_penalty = max(0, (battery_soc_min - potential_battery_soc) ** 2)
    overcharge_penalty = max(0, (potential_battery_soc - battery_soc_max) ** 2)

    soc_penalty = overdischarge_penalty + overcharge_penalty

    reward_sell = (solar_to_grid + battery_to_grid) * max(state.predicted_prices[0], 0) * (1 - soc_penalty)
    reward_buy = grid_to_battery * min(state.predicted_prices[0], 0) * (1 - soc_penalty)

    # Time-of-day adjustments
    if 0 <= state.hour_of_day < 17 or 21 <= state.hour_of_day < 24:
        reward_sell *= 0.85 if reward_sell > 0 else 1.15
        reward_buy *= 0.95 if reward_buy > 0 else 1.05
    else:
        reward_sell *= 1.30 if reward_sell > 0 else 0.70
        reward_buy *= 0.60 if reward_buy > 0 else 1.40

    return reward_sell + reward_buy

def simulate(node, depth):
    current_state = node.state
    step = 0
    total_reward = 0
    while not current_state.is_terminal(step, depth):
        action = random.choice(node.untried_actions)
        current_state = transition(current_state, action)
        reward = get_reward(action, current_state)
        total_reward += reward
        step += 1
    return total_reward

def parallel_mcts(root, iterations=100, depth=50):
    for _ in range(iterations):
        node = root
        path = []
        while node.children:
            node = node.select_child()
            path.append(node)

        if node.untried_actions:
            action = node.untried_actions.pop()
            new_state = transition(node.state, action)
            child_node = Node(new_state, node)
            node.children.append(child_node)
            node = child_node

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(simulate, node, depth) for _ in range(4)]
            results = [f.result() for f in futures]

        total_simulated_reward = sum(results)
        for n in path:
            n.update(total_simulated_reward / len(path))

# Example usage
initial_state = State(battery_soc=50, predicted_prices=[10, 9, 8, 7, 6], hour_of_day=12)
root = Node(initial_state)
parallel_mcts(root, iterations=10, depth=10)
