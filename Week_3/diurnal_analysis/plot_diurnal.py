import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data you just calculated
df = pd.read_csv("diurnal_pm2.5_seasonwise.csv")

# Set the style
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Create the line plot
# We use a custom order for seasons to make sense of the plot
season_order = ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']
sns.lineplot(data=df, x='hour', y='PM 2.5', hue='season', 
             hue_order=season_order, marker='o', linewidth=2.5)

# Add titles and labels
plt.title("Diurnal Variation of PM 2.5 in Delhi (2021-2024)", fontsize=16, fontweight='bold')
plt.xlabel("Hour of the Day (24-Hour Format)", fontsize=12)
plt.ylabel("Average PM 2.5 Concentration (µg/m³)", fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Highlight the peaks with shading
plt.axvspan(7, 10, color='gray', alpha=0.1, label="Morning Peak")
plt.axvspan(19, 23, color='gray', alpha=0.1, label="Night Peak")

plt.legend(title="Season", loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Save for the PPT
plt.savefig("diurnal_trends_comparison.png", dpi=300)
print("Plot saved as diurnal_trends_comparison.png")
plt.show()