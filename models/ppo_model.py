import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class TFTAgent:
    def __init__(self, env_id="TFT-v0", model_path="ppo_tft_agent"):
        self.env_id = env_id
        self.model_path = model_path
        self.model = None

    def setup_env(self):
        env = gym.make(self.env_id)
        return DummyVecEnv([lambda: env])

    def train(self, timesteps=int(1e6)):
        env = self.setup_env()
        self.model = PPO("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=timesteps)

    def save(self):
        if self.model is not None:
            self.model.save(self.model_path)
        else:
            print("No model to save. Train the model first.")

    def load(self):
        env = self.setup_env()
        self.model = PPO.load(self.model_path, env=env)

    def evaluate(self, steps=1000):
        if self.model is None:
            print("No model to evaluate. Load or train a model first.")
            return

        env = self.setup_env()
        obs = env.reset()
        for _ in range(steps):
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
