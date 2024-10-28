from model import TradingEnv, create_model
from utils import load_data

def train_model(data_path, timesteps=10000):
    data = load_data(data_path)
    env = TradingEnv(data)
    model = create_model(env)
    model.learn(total_timesteps=timesteps)
    model.save('models/dqn_trade_model')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    data_path = "data/merged_bid_ask_ohlcv_data.csv"
    train_model(data_path)
