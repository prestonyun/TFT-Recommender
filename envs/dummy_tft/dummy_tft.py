import numpy as np
from enum import Enum

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