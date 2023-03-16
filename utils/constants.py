# enums.py

from enum import Enum

class Constants(Enum):
    BOARD_SIZE = 21,
    BENCH_SIZE = 11,
    SHOP_SIZE = 5,
    MAX_CHAMPION_ID = 100,
    MAX_CHAMPION_COST = 10,
    MAX_CHAMPION_LEVEL = 3,
    MAX_ITEMS_PER_CHAMPION = 6,
    MAX_ITEM_ID = 100,
    MAX_LEVEL = 10,
    MAX_HEALTH = 100,
    MAX_EXPERIENCE = 100,
    MAX_GOLD = 500,
    MAX_EXPERIENCE_TO_NEXT_LEVEL = 100,
    NUM_ACTIONS = 100,
    NUM_OPPONENTS = 7

class GameBoard(Enum):
    # Game board positions (top left is 0, middle left is 7, bottom left is 14)
    POSITION_0 = 0,
    POSITION_1 = 1,
    POSITION_2 = 2,
    POSITION_3 = 3,
    POSITION_4 = 4,
    POSITION_5 = 5,
    POSITION_6 = 6,
    POSITION_7 = 7,
    POSITION_8 = 8,
    POSITION_9 = 9,
    POSITION_10 = 10,
    POSITION_11 = 11,
    POSITION_12 = 12,
    POSITION_13 = 13,
    POSITION_14 = 14,
    POSITION_15 = 15,
    POSITION_16 = 16,
    POSITION_17 = 17,
    POSITION_18 = 18,
    POSITION_19 = 19,
    POSITION_20 = 20
    
class SynergyTiers(Enum):
    # Synergies
    ASSASSINS = 3,
    BLADEMASTER = 3,
    BRAWLERS = 3,
    DEMON = 3,
    DRAGON = 3,
    ELEMENTALIST = 3,
    GUARDIAN = 3,
    KNIGHT = 3,
    MAGE = 3,
    SHAPESHIFTER = 3,
    SORCERER = 3,
    YORDLE = 3