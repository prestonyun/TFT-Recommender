from gymnasium.envs.registration import register

from .tft_gym_env import *
from .env import *

register(
    id='envs/tft_gym_env:TFTGymEnv',
    entry_point='envs.tft_gym_env:TFTGymEnv',
    max_episode_steps=1000,
)