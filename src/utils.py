import pandas as pd
import numpy as np

class Benchmark:
    def __init__(self, data):
        self.data = data

    def get_twap_trades(self, initial_inventory, timeframe=390):
        twap_shares_per_step = initial_inventory / timeframe
        trades = []
        remaining_inventory = initial_inventory

        for step, row in self.data.iterrows():
            size_of_slice = min(twap_shares_per_step, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trades.append({
                'timestamp': row['timestamp'],
                'price': row['close'],
                'shares': size_of_slice,
                'inventory': remaining_inventory,
            })

        return pd.DataFrame(trades)

    def get_vwap_trades(self, initial_inventory, timeframe=390):
        total_volume = self.data['volume'].sum()
        trades = []
        remaining_inventory = initial_inventory

        for step, row in self.data.iterrows():
            volume_ratio = row['volume'] / total_volume
            size_of_slice = min(volume_ratio * initial_inventory, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trades.append({
                'timestamp': row['timestamp'],
                'price': row['close'],
                'shares': size_of_slice,
                'inventory': remaining_inventory,
            })

        return pd.DataFrame(trades)

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_total_cost(trades):
    return sum(trades['shares'] * trades['price'])
