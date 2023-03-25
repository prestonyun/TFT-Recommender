import gymnasium as gym
from gym import spaces
import numpy as np
from utils import Items, Champions, Actions

# TODO: # 'Combat Mechanics': higher power level wins, if equal, then the one with the higher cost wins
        # Define actions
        # Level up mechanic
        # Reward function
        # Restore board/bench state after 'combine_champions' action


class DummyTFTEnv(gym.Env):
    def __init__(self):
        super(DummyTFTEnv, self).__init__()

        self.board_size = 5
        self.bench_size = 3
        self.shop_size = 2

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
        stronger_champion = None
        if num_copies[Champions.CHAMPION1STAR1COST] == 3:
            stronger_champion = Champions.CHAMPION2STAR1COST
            all_owned_champs.remove(Champions.CHAMPION1STAR1COST)
            all_owned_champs.remove(Champions.CHAMPION1STAR1COST)
            all_owned_champs.remove(Champions.CHAMPION1STAR1COST)
        elif num_copies[Champions.CHAMPION1STAR2COST] == 3:
            stronger_champion = Champions.CHAMPION2STAR2COST
            all_owned_champs.remove(Champions.CHAMPION1STAR2COST)
            all_owned_champs.remove(Champions.CHAMPION1STAR2COST)
            all_owned_champs.remove(Champions.CHAMPION1STAR2COST)
        elif num_copies[Champions.CHAMPION1STAR3COST] == 3:
            stronger_champion = Champions.CHAMPION2STAR3COST
            all_owned_champs.remove(Champions.CHAMPION1STAR3COST)
            all_owned_champs.remove(Champions.CHAMPION1STAR3COST)
            all_owned_champs.remove(Champions.CHAMPION1STAR3COST)
        elif num_copies[Champions.CHAMPION2STAR1COST] == 3:
            stronger_champion = Champions.CHAMPION3STAR1COST
            all_owned_champs.remove(Champions.CHAMPION2STAR1COST)
            all_owned_champs.remove(Champions.CHAMPION2STAR1COST)
            all_owned_champs.remove(Champions.CHAMPION2STAR1COST)
        elif num_copies[Champions.CHAMPION2STAR2COST] == 3:
            stronger_champion = Champions.CHAMPION3STAR2COST
            all_owned_champs.remove(Champions.CHAMPION2STAR2COST)
            all_owned_champs.remove(Champions.CHAMPION2STAR2COST)
            all_owned_champs.remove(Champions.CHAMPION2STAR2COST)
        elif num_copies[Champions.CHAMPION2STAR3COST] == 3:
            stronger_champion = Champions.CHAMPION3STAR3COST
            all_owned_champs.remove(Champions.CHAMPION2STAR3COST)
            all_owned_champs.remove(Champions.CHAMPION2STAR3COST)
            all_owned_champs.remove(Champions.CHAMPION2STAR3COST)
        
        if stronger_champion:
            all_owned_champs.append(stronger_champion)
            return self.combine_champions(all_owned_champs)
        return all_owned_champs

    def step(self, action):
        action_type, shop_idx, board_idx = action

        if action_type == 0:  # Buy champion
            champion = self.champions[self.observation['shop'][shop_idx]]
            self.observation['board'][board_idx] = champion['power_level']

        elif action_type == 1:  # Place champion
            self.observation['board'][board_idx] = self.observation['board'][shop_idx]

        # Determine the winner of the fight based on the total power level of the boards
        player_power_level = np.sum(self.observation['board'])
        opponent_power_level = np.random.randint(1, 11)  # random power level for the opponent's board
        if player_power_level > opponent_power_level:
            reward = 1  # player wins
        elif player_power_level < opponent_power_level:
            reward = -1  # player loses
        else:
            reward = 0  # draw

        # Check if the game is done (board is full)
        done = not np.any(self.observation['board'] == 0)

        # Update the shop
        self.observation['shop'] = np.random.randint(0, len(self.champions), size=self.shop_size)

        # Calculate reward based on the total power level on the board
        reward = np.sum(self.observation['board'])

        return self.observation, reward, done, {}

    def reset(self):
        self.observation = {
            'board': np.zeros(self.board_size, dtype=np.float32),
            'shop': np.random.randint(0, len(self.champions), size=self.shop_size),
        }
        return self.observation
