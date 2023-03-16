import gymnasium as gym
from gym import spaces
from enum import Enum
import numpy as np
import random

# TODO: # 'Combat Mechanics': higher power level wins, if equal, then the one with the higher cost wins
        # Define actions
        # Level up mechanic
        # Reward function


class Items(Enum):
    ITEM1 = {'power_level': 1, 'id': 1}
    ITEM2 = {'power_level': 2, 'id': 2}
    ITEM3 = {'power_level': 3, 'id': 3}
    ITEM11 = {'power_level': 2, 'id': 4}
    ITEM12 = {'power_level': 3, 'id': 5}
    ITEM13 = {'power_level': 4, 'id': 6}
    ITEM22 = {'power_level': 4, 'id': 7}
    ITEM23 = {'power_level': 5, 'id': 8}
    ITEM33 = {'power_level': 6, 'id': 9}

class Champions(Enum):
    CHAMPION1STAR1COST = {'power_level': 1, 'id': 1, 'cost': 1}
    CHAMPION1STAR2COST = {'power_level': 2, 'id': 2, 'cost': 2}
    CHAMPION1STAR3COST = {'power_level': 3, 'id': 3, 'cost': 3}
    CHAMPION2STAR1COST = {'power_level': 2, 'id': 4, 'cost': 2}
    CHAMPION2STAR2COST = {'power_level': 3, 'id': 5, 'cost': 5}
    CHAMPION2STAR3COST = {'power_level': 4, 'id': 6, 'cost': 8}
    CHAMPION3STAR1COST = {'power_level': 3, 'id': 7, 'cost': 8}
    CHAMPION3STAR2COST = {'power_level': 4, 'id': 8, 'cost': 17}
    CHAMPION3STAR3COST = {'power_level': 5, 'id': 9, 'cost': 26}

class Actions(Enum):
    # 0: Buy champion, 1: Sell champion, 2: Buy experience, 3: Place item, 4: Move champion, 5: Skip
    BUY_CHAMPION = 0
    SELL_CHAMPION = 1
    BUY_EXPERIENCE = 2
    PLACE_ITEM = 3
    MOVE_CHAMPION = 4
    SKIP = 5


class DummyTFTEnv(gym.Env):
    def __init__(self):
        super(DummyTFTEnv, self).__init__()

        self.board_size = 5
        self.bench_size = 3
        self.shop_size = 3

        self.max_champion_id = 9
        self.max_item_id = 6

        self.board_items = []

        self.max_actions = 10

        self.observation_space = spaces.Dict({
            'player_gold': spaces.Discrete(100),
            'player_level': spaces.Discrete(5),

            "player_board": spaces.Tuple([spaces.Dict({
                "champion_id": spaces.Discrete(9),
                "item_ids": spaces.Box(low=0, high=9, shape=(2,), dtype=int)
            })] * self.board_size),
            "player_bench": spaces.Tuple([spaces.Dict({
                "champion_id": spaces.Discrete(9),
                "item_ids": spaces.Box(low=0, high=9, shape=(2,), dtype=int)
            })] * self.bench_size),
            'shop': spaces.Tuple([spaces.Dict({
                "champion_id": spaces.Discrete(9),
            })] * self.shop_size),
        })

        self.action_space = spaces.Tuple((
            spaces.Discrete(3),  # 0: Buy champion, 1: Place champion, 2: Skip
            spaces.Discrete(self.shop_size),  # Index of the champion in the shop
            spaces.Discrete(self.board_size),  # Index of the board position
        ))

    def attach_item(self, item, champion):
        if len(champion['item_ids']) < 2:
            champion['item_ids'].append(item['id'])
            return True
        return False

    def combine_items(self, item1, item2):
        if item1 == Items.ITEM1 and item2 == Items.ITEM1:
            return Items.ITEM11
        elif item1 == Items.ITEM2 and item2 == Items.ITEM2:
            return Items.ITEM22
        elif item1 == Items.ITEM3 and item2 == Items.ITEM3:
            return Items.ITEM33
        elif (item1 == Items.ITEM1 and item2 == Items.ITEM2) or (item1 == Items.ITEM2 and item2 == Items.ITEM1):
            return Items.ITEM12
        elif (item1 == Items.ITEM1 and item2 == Items.ITEM3) or (item1 == Items.ITEM3 and item2 == Items.ITEM1):
            return Items.ITEM13
        elif (item1 == Items.ITEM2 and item2 == Items.ITEM3) or (item1 == Items.ITEM3 and item2 == Items.ITEM2):
            return Items.ITEM23
        return None
        
    def combine_champions(self, all_owned_champs):
        num_copies = {i:all_owned_champs.count(i) for i in all_owned_champs}
        if num_copies[Champions.CHAMPION1STAR1COST] == 3:
            stronger_champion = Champions.CHAMPION2STAR1COST
        elif num_copies[Champions.CHAMPION1STAR2COST] == 3:
            stronger_champion = Champions.CHAMPION2STAR2COST
        elif num_copies[Champions.CHAMPION1STAR3COST] == 3:
            stronger_champion = Champions.CHAMPION2STAR3COST
        elif num_copies[Champions.CHAMPION2STAR1COST] == 3:
            stronger_champion = Champions.CHAMPION3STAR1COST
        elif num_copies[Champions.CHAMPION2STAR2COST] == 3:
            stronger_champion = Champions.CHAMPION3STAR2COST
        elif num_copies[Champions.CHAMPION2STAR3COST] == 3:
            stronger_champion = Champions.CHAMPION3STAR3COST
        else:
            return all_owned_champs
        
        # Remove the 3 weaker champions from the original array

        #for i in range(len(all_owned_champs)):
            
        # Add the stronger champion to the array
        all_owned_champs.append(stronger_champion)
        return all_owned_champs


    def step(self, action):
        action_type, shop_idx, board_idx = action

        if action_type == 0:  # Buy champion
            champion = self.champions[self.observation['shop'][shop_idx]]
            self.observation['board'][board_idx] = champion['power_level']

        elif action_type == 1:  # Place champion
            self.observation['board'][board_idx] = self.observation['board'][shop_idx]

        # Calculate reward based on the total power level on the board
        reward = np.sum(self.observation['board'])

        # Check if the game is done (board is full)
        done = not np.any(self.observation['board'] == 0)

        # Update the shop
        self.observation['shop'] = np.random.randint(0, len(self.champions), size=self.shop_size)

        return self.observation, reward, done, {}

    def reset(self):
        self.observation = {
            'board': np.zeros(self.board_size, dtype=np.float32),
            'shop': np.random.randint(0, len(self.champions), size=self.shop_size),
        }
        return self.observation
