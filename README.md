# World Population Analysis - PRODIGY_DS_01

## Objective

Analyze and visualize world population data to understand population distribution across income groups and countries. Create clear, professional visualizations suitable for data-driven decision making.

## Dataset

**Source:** World Bank - Population Indicators (API_SP.POP.TOTL)

**Contents:**
- Population data for 266 countries and regions (1960-2024)
- Country metadata including income group classifications
- Annual population figures in people

**Income Groups:**
- Low income
- Lower middle income
- Upper middle income
- High income

## Visualizations

### 1. Population by Income Group (2024)
**File:** `population_by_income_group_2024.png`
- Bar chart showing total population for each income group
- Values displayed in millions
- Color-coded for easy comparison

### 2. Population Distribution - Linear Scale
**File:** `population_distribution_linear.png`
- Histogram of population values across all countries
- Shows the raw distribution of country populations
- Useful for identifying outliers and clustering

### 3. Population Distribution - Log Scale
**File:** `population_distribution_log.png`
- Histogram using logarithmic (log₁₀) scale
- Reveals patterns across wide population ranges
- Better visibility of smaller countries

## Tools Used

- **Python 3.x**
- **Libraries:**
  - `pandas` - Data manipulation and grouping
  - `matplotlib` - Visualization and plotting
  - `seaborn` - Enhanced styling
  - `numpy` - Numerical operations

## Installation

1. **Install Python packages:**
   ```bash
   pip install pandas matplotlib seaborn numpy
   ```

2. **Verify installation:**
   ```bash
   python -m pip list
   ```

## How to Run

1. **Navigate to the project directory:**
   ```bash
   cd c:\PRODIGY_DS_01
   ```

2. **Run the visualization script:**
   ```bash
   python task1_visualization.py
   ```

3. **Output:**
   - Three PNG files will be generated in the project folder
   - Combined visualization will display on screen

## Project Structure

```
PRODIGY_DS_01/
├── task1_visualization.py          # Main analysis script
├── README.md                        # Project documentation
├── data/
│   ├── API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv
│   ├── Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv
│   └── Metadata_Indicator_API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv
├── population_by_income_group_2024.png
├── population_distribution_linear.png
└── population_distribution_log.png
```

## Key Insights

- Shows how global population is distributed across economic income levels
- Linear and log scales reveal different patterns in the data
- Color-coded visualizations make comparisons intuitive and accessible

## Author

Created for PRODIGY Data Science Internship - Task 1

## License

World Bank data used for educational purposes