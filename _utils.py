#import cv2
import luadata

def get_shop_state():
    shop_state = []
    return shop_state

def read_champions():
    champions = luadata.read("champions.lua", encoding="utf-8")
    return champions