
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data = pd.read_csv('5.3.data/metanav.csv')

# Sort by global_step to ensure lines connect in order
data = data.sort_values('global_step')

plt.figure(figsize=(10, 6), dpi=150)

# Collect all columns ending with 'inner_action_loss'
loss_columns = [col for col in data.columns if col.endswith('losses/inner_action_loss')]
all_values = data[loss_columns].values

average_values = np.mean(all_values, axis=1)
std_values = np.std(all_values, axis=1)

plt.plot(data['global_step'], average_values, label='Average Inner Action Loss', color='blue', linewidth=2, marker='o')
plt.fill_between(data['global_step'], average_values - std_values, average_values + std_values, color='blue', alpha=0.3)

plt.xlabel('Tasks', fontsize=18)
plt.ylabel('Inner Action Loss', fontsize=18)
plt.title('Average Inner Action Loss vs. Tasks', fontsize=18)

plt.gca().set_axisbelow(True)
plt.grid(True, linestyle='--', color='grey', alpha=0.20, which='both', axis='both')
plt.tight_layout()
plt.savefig('plots/meta_nav_losses.png')

plt.show()


