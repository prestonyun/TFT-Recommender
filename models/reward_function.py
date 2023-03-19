# reward_function.py

class RewardFunction:
    def __init__(self, weights):
        self.weights = weights

    def round_result_reward(self, round_result):
        pass

    def economy_reward(self, gold, streak):
        pass

    def synergy_reward(self, active_synergies, inactive_synergies):
        pass

    def champion_level_reward(self, champion_level):
        pass

    def time_step_reward(self):
        # Reward for taking a time step (encourage quicker decision-making)
        pass

    def compute_reward(self, round_result, gold, streak, active_synergies, champion_levels):
        reward = 0
        reward += self.weights['round_result'] * self.round_result_reward(round_result)
        reward += self.weights['economy'] * self.economy_reward(gold, streak)
        reward += self.weights['synergy'] * self.synergy_reward(active_synergies)
        reward += self.weights['champion_level'] * self.champion_level_reward(champion_levels)
        reward += self.weights['time_step'] * self.time_step_reward()
        return reward