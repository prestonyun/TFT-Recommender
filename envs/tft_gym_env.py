import gymnasium as gym
from gym import spaces
from utils.constants import Constants, SynergyTiers

single_action_space = spaces.Dict({
    'action_type': spaces.Discrete(n_action_types),
    'params': spaces.Tuple((spaces.Discrete(n_champions), spaces.Discrete(n_positions), spaces.Discrete(n_items))),
})

class TFTGymEnv(gym.Env):
    def __init__(self):
        OPPONENT_FEATURES = None
        self.action_space = spaces.Tuple([single_action_space for _ in range(max_actions)])
        self.observation_space = spaces.Dict({
            'health': spaces.Discrete(100),
            'level': spaces.Discrete(10),
            'experience': spaces.Discrete(100),
            'experience_to_next_level': spaces.Discrete(100),
            'gold': spaces.Discrete(500),
            'active_synergies': spaces.Dict({synergy: spaces.Discrete(max_tier + 1) for synergy, max_tier in SynergyTiers.__members__.items()}),
            "opponents": spaces.Box(low=0, high=1, shape=(Constants.NUM_OPPONENTS, OPPONENT_FEATURES), dtype=float),
            "shop": spaces.Tuple([spaces.Dict({
                "champion_id": spaces.Discrete(Constants.MAX_CHAMPION_ID),
                "champion_cost": spaces.Discrete(Constants.MAX_CHAMPION_COST)
            })] * Constants.SHOP_SIZE),
            "player_board": spaces.Tuple([spaces.Dict({
                "champion_id": spaces.Discrete(Constants.MAX_CHAMPION_ID),
                "champion_level": spaces.Discrete(Constants.MAX_CHAMPION_LEVEL),
                "item_ids": spaces.Box(low=0, high=Constants.MAX_ITEM_ID, shape=(Constants.MAX_ITEMS_PER_CHAMPION,), dtype=int)
            })] * Constants.BOARD_SIZE),
            "player_bench": spaces.Tuple([spaces.Dict({
                "champion_id": spaces.Discrete(Constants.MAX_CHAMPION_ID),
                "champion_level": spaces.Discrete(Constants.MAX_CHAMPION_LEVEL),
                "item_ids": spaces.Box(low=0, high=Constants.MAX_ITEM_ID, shape=(Constants.MAX_ITEMS_PER_CHAMPION,), dtype=int)
            })] * Constants.BENCH_SIZE),
        })


    def step(self, actions):
        # Apply the actions
        for action in actions:
            action_type, params = action['action_type'], action['params']
            if action_type == 0:
                pass

        # Calculate the reward
        reward = ...

        # Check if the episode has ended
        done = ...

        # Return the new state, reward, and 'done' flag
        return self.get_observation(), reward, done, {}


    def reset(self):
        # Initialize opponents
        # Initialize shop, board, bench, player
        return 0

    def render(self, mode='human', close=False):
        pass

    def get_observation(self):
        pass

    def observe_opponents(self):
        OPPONENT_FEATURES = None
        return OPPONENT_FEATURES