import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# Set aesthetic style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Calibri', 'Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linestyle'] = '--'

# Load the dataset
df = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv', skiprows=4)

# Load metadata to get income group information
metadata = pd.read_csv('data/Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv')
metadata = metadata[['Country Code', 'IncomeGroup']].rename(columns={'Country Code': 'Country Code'})

# Merge main data with metadata to get income groups
df = df.merge(metadata, left_on='Country Code', right_on='Country Code', how='left')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Select numeric columns (years) for analysis
year_columns = [col for col in df.columns if col.isdigit()]
numeric_data = df[year_columns].astype(float)

# Create a bar chart for top 10 countries by population (using 2024 data)
fig = plt.figure(figsize=(18, 12))
fig.suptitle('World Population Analysis - 2024', fontsize=18, fontweight='300', y=0.995, color='#1a1a1a', family='Segoe UI')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32, top=0.93, bottom=0.08, left=0.08, right=0.95)

# Bar chart (spans 2 rows on the left)
ax1 = fig.add_subplot(gs[:, 0])

# Filter data to only include actual income groups (exclude empty and aggregate rows)
valid_income_groups = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
df_income = df[df['IncomeGroup'].isin(valid_income_groups)].copy()

# Convert 2024 population to millions and group by income group
df_income['2024'] = pd.to_numeric(df_income['2024'], errors='coerce')
population_by_income = df_income.groupby('IncomeGroup')['2024'].sum() / 1_000_000

# Order income groups for better visualization
income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
population_by_income = population_by_income.reindex(income_order)

# Create the bar chart
colors = ['#2c3e50', '#3498db', '#2ecc71', '#f39c12']
bars = ax1.bar(range(len(population_by_income)), population_by_income.values, color=colors, edgecolor='white', linewidth=2, width=0.6)

# Customize the plot
ax1.set_xticks(range(len(population_by_income)))
ax1.set_xticklabels(population_by_income.index, rotation=45, ha='right', fontsize=11, fontweight='500', family='Segoe UI')
ax1.set_ylabel('Population (Millions)', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax1.set_title('World Population in 2024 - Share by Income Group', fontsize=14, fontweight='400', pad=20, color='#1a1a1a', family='Segoe UI')
ax1.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
ax1.set_axisbelow(True)

# Add value labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 100, f'{height:.0f}M', 
             ha='center', va='bottom', fontsize=11, fontweight='400', color='#1a1a1a', family='Segoe UI', style='italic')

# Improve tick labels
ax1.tick_params(axis='both', labelsize=10, colors='#1a1a1a')
ax1.set_ylim(0, max(population_by_income.values) * 1.12)
for spine in ax1.spines.values():
    spine.set_edgecolor('#bdc3c7')
    spine.set_linewidth(1.2)

# Create separate figure for bar chart and save
fig_bar = plt.figure(figsize=(10, 7))
ax_bar = fig_bar.add_subplot(111)
bars = ax_bar.bar(range(len(population_by_income)), population_by_income.values, color=colors, edgecolor='white', linewidth=2, width=0.6)
ax_bar.set_xticks(range(len(population_by_income)))
ax_bar.set_xticklabels(population_by_income.index, rotation=45, ha='right', fontsize=11, fontweight='500', family='Segoe UI')
ax_bar.set_ylabel('Population (Millions)', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax_bar.set_title('World Population in 2024 - Share by Income Group', fontsize=14, fontweight='400', pad=20, color='#1a1a1a', family='Segoe UI')
ax_bar.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
ax_bar.set_axisbelow(True)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, height + 100, f'{height:.0f}M', 
             ha='center', va='bottom', fontsize=11, fontweight='400', color='#1a1a1a', family='Segoe UI', style='italic')
ax_bar.tick_params(axis='both', labelsize=10, colors='#1a1a1a')
ax_bar.set_ylim(0, max(population_by_income.values) * 1.12)
for spine in ax_bar.spines.values():
    spine.set_edgecolor('#bdc3c7')
    spine.set_linewidth(1.2)
fig_bar.tight_layout()
fig_bar.savefig('population_by_income_group_2024.png', dpi=300, bbox_inches='tight')
print("Saved: population_by_income_group_2024.png")
plt.close(fig_bar)

# Linear histogram (top right)
ax2 = fig.add_subplot(gs[0, 1])
populations_2024 = (df['2024'].astype(float) / 1_000_000).dropna()  # Convert to millions
n, bins, patches = ax2.hist(populations_2024, bins=30, edgecolor='white', alpha=0.85, color='#3498db', linewidth=0.5)

# Add gradient to histogram
cm = plt.cm.Blues
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(0.4 + 0.6 * (i / len(patches))))

ax2.set_xlabel('Population (Millions)', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax2.set_ylabel('Number of Countries', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax2.set_title('Population Distribution - Linear Scale', fontsize=14, fontweight='400', pad=20, color='#1a1a1a', family='Segoe UI')
ax2.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
ax2.set_axisbelow(True)
ax2.tick_params(axis='both', labelsize=10, colors='#34495e')
for spine in ax2.spines.values():
    spine.set_edgecolor('#bdc3c7')
    spine.set_linewidth(1.2)

# Create separate figure for linear histogram and save
fig_linear = plt.figure(figsize=(10, 7))
ax_linear = fig_linear.add_subplot(111)
n, bins, patches = ax_linear.hist(populations_2024, bins=30, edgecolor='white', alpha=0.85, color='#3498db', linewidth=0.5)
cm = plt.cm.Blues
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(0.4 + 0.6 * (i / len(patches))))
ax_linear.set_xlabel('Population (Millions)', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax_linear.set_ylabel('Number of Countries', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax_linear.set_title('Population Distribution - Linear Scale', fontsize=14, fontweight='400', pad=20, color='#1a1a1a', family='Segoe UI')
ax_linear.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
ax_linear.set_axisbelow(True)
ax_linear.tick_params(axis='both', labelsize=10, colors='#34495e')
for spine in ax_linear.spines.values():
    spine.set_edgecolor('#bdc3c7')
    spine.set_linewidth(1.2)
fig_linear.tight_layout()
fig_linear.savefig('population_distribution_linear.png', dpi=300, bbox_inches='tight')
print("Saved: population_distribution_linear.png")
plt.close(fig_linear)

# Log10 histogram (bottom right)
ax3 = fig.add_subplot(gs[1, 1])
# Filter out zero and negative values for log scale
populations_log = (df['2024'].astype(float) / 1_000_000).dropna()
populations_log = populations_log[populations_log > 0]  # Keep only positive values
log_populations = np.log10(populations_log)
n, bins, patches = ax3.hist(log_populations, bins=30, edgecolor='white', alpha=0.85, color='#2ecc71', linewidth=0.5)

# Add gradient to histogram
cm = plt.cm.Greens
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(0.4 + 0.6 * (i / len(patches))))

ax3.set_xlabel('Population (log₁₀ Millions)', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax3.set_ylabel('Number of Countries', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax3.set_title('Population Distribution - Log Scale', fontsize=14, fontweight='400', pad=20, color='#1a1a1a', family='Segoe UI')
ax3.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
ax3.set_axisbelow(True)
ax3.tick_params(axis='both', labelsize=10, colors='#34495e')
for spine in ax3.spines.values():
    spine.set_edgecolor('#bdc3c7')
    spine.set_linewidth(1.2)

# Create separate figure for log histogram and save
fig_log = plt.figure(figsize=(10, 7))
ax_log = fig_log.add_subplot(111)
n, bins, patches = ax_log.hist(log_populations, bins=30, edgecolor='white', alpha=0.85, color='#2ecc71', linewidth=0.5)
cm = plt.cm.Greens
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(0.4 + 0.6 * (i / len(patches))))
ax_log.set_xlabel('Population (log₁₀ Millions)', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax_log.set_ylabel('Number of Countries', fontsize=12, fontweight='500', color='#1a1a1a', family='Segoe UI')
ax_log.set_title('Population Distribution - Log Scale', fontsize=14, fontweight='400', pad=20, color='#1a1a1a', family='Segoe UI')
ax_log.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
ax_log.set_axisbelow(True)
ax_log.tick_params(axis='both', labelsize=10, colors='#34495e')
for spine in ax_log.spines.values():
    spine.set_edgecolor('#bdc3c7')
    spine.set_linewidth(1.2)
fig_log.tight_layout()
fig_log.savefig('population_distribution_log.png', dpi=300, bbox_inches='tight')
print("Saved: population_distribution_log.png")
plt.close(fig_log)

# Display the combined figure
print("\nDisplaying combined visualization...")
plt.tight_layout()
plt.show()
