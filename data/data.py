import pandas as pd

data = pd.read_csv('train_data.csv')
data['out_pressure'] = 0.7
data.to_csv('train_data_.csv', index=False)

data = pd.read_csv('test_data.csv')
data['out_pressure'] = 0.7
data.to_csv('test_data_.csv', index=False)