#import cv2
import luadata

def get_game_state():
    game_state = {
        "gold": 0,
        "shop_state": [],
        "bench_state": [],
        "battlefield_state": [[None, None, None], [None, None, None], [None, None, None]]
    }
    return game_state

def get_shop_state():
    shop_state = []
    return shop_state

def read_champions():
    champions = luadata.read("champions.lua", encoding="utf-8")
    return champions