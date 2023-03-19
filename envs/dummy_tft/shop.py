import numpy as np
import random
from utils import Champions

class Shop:
    def __init__(self, player_level, shop_size=2):
        self.player_level = player_level
        self.shop_size = shop_size
        self.champion_pool = self.generate_champion_pool()

    def generate_champion_pool(self):
        probabilities = {
            1: [1.0, 0.0, 0.0],
            2: [0.75, 0.25, 0.0],
            3: [0.50, 0.375, 0.125],
            # Add more levels as needed
        }
        
        pool = []
        for i in range(self.shop_size):
            champion_tier = np.random.choice([1, 2, 3], p=probabilities[self.player_level])
            champion = self.get_champion_by_tier(champion_tier)
            pool.append(champion)
        return pool

    @staticmethod
    def get_champion_by_tier(tier):
        champions_by_tier = {
            1: [Champions.CHAMPION1STAR1COST],
            2: [Champions.CHAMPION1STAR2COST],
            3: [Champions.CHAMPION1STAR3COST],
            # Add more tiers as needed
        }
        return random.choice(champions_by_tier[tier])

# Example usage:
shop = Shop(player_level=3)
print([champion.name for champion in shop.champion_pool])
