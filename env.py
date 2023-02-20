import _utils

class TFTGameState:
    def __init__(self, bench_state, battlefield_state, shop_state, items, health, gold, experience, experience_to_next_level, level, opponents):
        self.bench_state = bench_state
        self.battlefield_state = battlefield_state
        self.shop_state = shop_state
        self.items = items
        self.health = health
        self.gold = gold
        self.experience = experience
        self.experience_to_next_level = experience_to_next_level
        self.level = level
        self.opponents = opponents

        self.champion_stats = _utils.read_champions()


class TFTGameActions:
    @staticmethod
    def buy_champion(game_state, champion):
        if champion.num_copies >= 2:
            if champion in game_state.shop_state and game_state.gold >= champion.cost:
                champion.starup()
                game_state.gold -= champion.cost
                return True
        elif champion in game_state.shop_state and game_state.gold >= champion.cost and len(game_state.bench_state) < 9:
            game_state.shop_state.remove(champion)
            game_state.bench_state.append(champion)
            game_state.gold -= champion.cost
            return True
        else:
            return False

    @staticmethod
    def sell_champion(game_state, champion):
        if champion in game_state.bench_state:
            game_state.bench_state.remove(champion)
            game_state.gold += champion.cost
        elif champion in game_state.battlefield_state:
            game_state.battlefield_state.remove(champion)
            game_state.gold += champion.cost
        
        if len(champion.items) > 0:
            for item in champion.items:
                game_state.items.append(champion.items.pop(item))
        return True

    @staticmethod
    def move_champion(game_state, champion, destination_row, destination_column):
        if champion in game_state.bench_state:
            game_state.bench_state.remove(champion)
            game_state.battlefield_state[destination_row][destination_column] = champion
        elif champion in game_state.battlefield_state:
            if game_state.battlefield_state[destination_row][destination_column] is None:
                for row_index, row in enumerate(game_state.battlefield_state):
                    for col_index, col in enumerate(row):
                        if col == champion:
                            game_state.battlefield_state[row_index][col_index] = None
                            game_state.battlefield_state[destination_row][destination_column] = champion
                            return True
            else:
                for row_index, row in enumerate(game_state.battlefield_state):
                    for col_index, col in enumerate(row):
                        if col == champion:
                            other_champion = game_state.battlefield_state[destination_row][destination_column]
                            game_state.battlefield_state[row_index][col_index] = other_champion
                            game_state.battlefield_state[destination_row][destination_column] = champion
                            return True
        return False


    @staticmethod
    def refresh_shop(game_state):
        if game_state.gold >= 2:
            game_state.gold -= 2
            game_state.shop_state = _utils.get_shop_state()
            return True
        else:
            return False

    @staticmethod
    def buy_experience(game_state):
        if game_state.gold >= 4 and game_state.level < 9:
            game_state.gold -= 4
            game_state.experience += 4
            if game_state.experience >= game_state.experience_to_next_level:
                game_state.level += 1
            return True
        else:
            return False

    @staticmethod
    def attach_item(game_state, champion, item):
        if champion in game_state.battlefield_state and len(champion.items) < 3 and item in game_state.items:
            champion.items.append(item)
            game_state.items.remove(item)
            return True
        else:
            return False

class TFTEnvironment:
    def __init__(self, game_state):
        self.game_state = game_state
        self.reward = 0
        
    def step(self, action):
        if action == "refresh_shop":
            self.game_state.refresh_shop()
        elif action == "buy_exp":
            self.game_state.buy_exp()
        elif action.startswith("buy_champ_"):
            champ_name = action.split("_")[2]
            self.game_state.buy_champion(champ_name)
        elif action.startswith("sell_champ_"):
            champ_name = action.split("_")[2]
            self.game_state.sell_champion(champ_name)
        elif action.startswith("move_champ_"):
            _, champ_name, destination_row, destination_column = action.split("_")
            destination_row, destination_column = int(destination_row), int(destination_column)
            self.game_state.move_champion(champ_name, destination_row, destination_column)
        elif action.startswith("buy_item_"):
            item_name = action.split("_")[2]
            self.game_state.buy_item(item_name)
        elif action.startswith("sell_item_"):
            item_name = action.split("_")[2]
            self.game_state.sell_item(item_name)
        
        self.reward = self.game_state.calculate_reward()
        observation = self.game_state.get_observation()
        done = self.game_state.is_terminal()
        
        return observation, self.reward, done
