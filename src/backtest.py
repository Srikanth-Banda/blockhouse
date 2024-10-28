import pandas as pd
from utils import Benchmark, calculate_total_cost
from model import TradingEnv, create_model
from utils import load_data

def run_backtest(data_path, model_path):
    # Load data
    data = load_data(data_path)

    # Initialize RL environment and load model
    env = TradingEnv(data)
    model = create_model(env)
    model.load(model_path)

    # Initialize Benchmark
    benchmark = Benchmark(data)

    # Generate TWAP and VWAP trades
    twap_trades = benchmark.get_twap_trades(initial_inventory=1000)
    vwap_trades = benchmark.get_vwap_trades(initial_inventory=1000)

    # Collect RL trades for comparison
    rl_trades = []
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, _, done, _ = env.step(action)
        rl_trades.append(obs)

    # Convert RL trades to DataFrame
    rl_trades_df = pd.DataFrame(rl_trades, columns=['close', 'volume', 'inventory', 'step', 'shares'])

    # Calculate costs
    twap_cost = calculate_total_cost(twap_trades)
    vwap_cost = calculate_total_cost(vwap_trades)
    rl_cost = calculate_total_cost(rl_trades_df)

    # Print the results
    print(f'TWAP Total Cost: {twap_cost}')
    print(f'VWAP Total Cost: {vwap_cost}')
    print(f'RL Model Total Cost: {rl_cost}')

if __name__ == "__main__":
    data_path = "data/merged_bid_ask_ohlcv_data.csv"
    model_path = "models/dqn_trade_model"
    run_backtest(data_path, model_path)
