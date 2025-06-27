import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('5.3.data/metanav.csv')
data = data.sort_values('global_step')

# Select relevant columns
loss_columns = [col for col in data.columns if col.endswith('losses/inner_action_loss')]
all_data = data[loss_columns]

# Compute mean and std across columns for each row
average_values = all_data.mean(axis=1)
std_values = all_data.std(axis=1)

# Smooth the data using a rolling window
window_size = 15
average_values = average_values.rolling(window=window_size, min_periods=1).mean()
std_values = std_values.rolling(window=window_size, min_periods=1).mean()

# Use the corresponding global_step values
global_steps = data.loc[all_data.index, 'global_step']

plt.figure(figsize=(10, 6), dpi=150)
plt.plot(global_steps, average_values, label='Average Action Loss') # , color='violet')
plt.fill_between(global_steps, average_values - std_values, average_values + std_values, color='gray', alpha=0.3)

plt.xlabel('Steps', fontsize=18)
plt.ylabel('Action Loss', fontsize=18)
# plt.title('Average Inner Action Loss ± Std vs. Tasks', fontsize=18)
plt.grid(True, linestyle='--', color='grey', alpha=0.20)
plt.xlim(0, 94000000)
# Change the ticks fontsize
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Put the scale in millions
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1e6)}M'))

plt.tight_layout()
plt.legend()
plt.savefig('plots/metanav_losses_avg_std.png')
plt.show()