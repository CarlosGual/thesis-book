import pandas as pd
import matplotlib.pyplot as plt

# Load the data (adjust the path if necessary)
df = pd.read_csv('icra_topics_2016_2025.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['robot_learning'], marker='o', label='Robot Learning')
plt.plot(df['year'], df['robot_navigation'], marker='o', label='Robot Navigation', color='violet')
plt.plot(df['year'], df['embodied_ai'], marker='o', label='Embodied AI', color='darkorange')

plt.xlabel('ICRA edition (year)', fontsize=18)
plt.ylabel('Approx. # accepted papers', fontsize=18)
# plt.title('Evolución de publicaciones en ICRA (2016–2025)')
plt.grid(True)
plt.ylim([0, 440])
plt.xlim([2016, 2025])
plt.grid(True, linestyle='--', color='grey', alpha=0.20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('Book/figures/introduction/icra_papers.pdf', bbox_inches='tight')
plt.show()
