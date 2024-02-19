import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Load the data
african_economies_df = pd.read_csv('african_economies_dataset.csv')

st.title("""
         # African Economies Analysis: GDP
         Focus on Ethiopia 
         """)

# Display the first few rows of the dataframe
st.write(african_economies_df.head())

# Create a multiindex setting the country as index and using the 'indicators' as columns
african_economies_df = african_economies_df.pivot_table(values='Value', index=('country','Date'), columns='indicator')

# Display the first few rows of the dataframe
st.write(african_economies_df.head())

# Filter the dataframe to only include data from 1995 onwards
date_mask = african_economies_df.index.get_level_values('Date') > 1994
african_economies_21st_df = african_economies_df.loc[date_mask]

african_economies_21st_clean_df = african_economies_21st_df.apply(lambda group: group.infer_objects(copy=False).interpolate(method='linear'))

for col in african_economies_21st_clean_df.columns:
    african_economies_21st_clean_df[col] = african_economies_21st_clean_df[col].replace(0, pd.NA)

african_economies_21st_clean_df = african_economies_21st_clean_df.drop(columns=['Oil exports, Value (Cur. USD)', 'External Public Debt from Private Creditors (Current USD)'])

# I create a boolean mask with the indexes to be dropped
drop_index_mask = african_economies_21st_clean_df.index.get_level_values(0).isin(['Algeria', 'Burundi', 'Central African Republic', 'Eritrea', 'Gambia',
       'Seychelles', 'Somalia', 'South Sudan'])
# I create a new dataframe wich contains the indexes to be dropped
drop_df = african_economies_21st_clean_df[drop_index_mask]

african_economies_21st_clean_df.drop(drop_df.index, inplace=True)

# Display the average increase of GDP per capita
african_groupby_mean_df = african_economies_21st_clean_df.groupby('country').mean()

# Sort the dataframe by GDP per capita and GDP growth
real_gdp_per_capita_growth_sorted = african_groupby_mean_df.sort_values(by='Annual growth rate of real GDP per capita (%)',ascending=False)
real_gdp_growth_sorted = african_groupby_mean_df.sort_values(by='Real GDP growth (annual %)',ascending=False)

# Display the top 10 countries by GDP per capita growth
st.write(real_gdp_per_capita_growth_sorted.head(10))

# Set the size of the figure
fig1 = plt.figure(figsize=(15,10))

# Barplot of the real GDP per capita growth
plt.bar(x=real_gdp_per_capita_growth_sorted.index, height=real_gdp_per_capita_growth_sorted['Annual growth rate of real GDP per capita (%)'])
plt.xticks(rotation=90)
plt.title('Average annual growth rate of real GDP per capita (%)')
plt.ylabel('Growth Rate (%)')
plt.xlabel('Country')
plt.show()

st.pyplot(fig1)

# Display the top 10 countries by GDP growth
st.write(real_gdp_growth_sorted.head(10))

# Plot the GDP growth trends for Ethiopia
fig2, ax = plt.subplots(2, 1, figsize=(12, 10))

ethiopia_mask = african_economies_21st_clean_df.index.get_level_values('country') == 'Ethiopia'
ethiopia_df = african_economies_21st_clean_df[ethiopia_mask]

# Subplot 1: Line plot
ax[0].plot(ethiopia_df.index.get_level_values('Date'), ethiopia_df['Real GDP growth (annual %)'], color='blue', label='Real GDP growth (annual %)', linestyle='-', marker='o')
ax[0].set_xlabel('Time (year)')
ax[0].set_ylabel('Real GDP growth (annual %)')
ax[0].set_title('Line Plot: GDP Growth Trends for Ethiopia')
ax[0].grid(linestyle='--', alpha=0.6)
ax[0].set_xticks(ethiopia_df.index.get_level_values('Date'))
ax[0].set_xticklabels(ethiopia_df.index.get_level_values('Date'), rotation=45)
ax[0].legend()

# Subplot 2: Bar chart
ax[1].bar(ethiopia_df.index.get_level_values('Date'), ethiopia_df['Real GDP growth (annual %)'], alpha=1, color='red', label='Real GDP growth (annual %)')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Real GDP growth (annual %)')
ax[1].set_title('Bar Plot: GDP Growth Trends for Ethiopia')
ax[1].grid(linestyle='--', alpha=0.6)
ax[1].set_xticks(ethiopia_df.index.get_level_values('Date'))
ax[1].set_xticklabels(ethiopia_df.index.get_level_values('Date'), rotation=45)
ax[1].legend()
plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()

st.pyplot(fig2)

# Plotting the Phillips curve
fig, ax = plt.subplots(figsize=(10, 6))

plt.scatter(ethiopia_df['Unemployment rate, (aged 15 over) (%)'],ethiopia_df['Inflation, consumer prices (annual %)'], color='blue',label='Unemployment Rate', marker='o')

plt.xlabel('Unemployment rate, (aged 15 over) (%)')
plt.ylabel('Inflation, consumer prices (annual %)')
plt.title('Phillips Curve: Correlation between Unemployment and Inflation')
plt.grid(True)

m, b = np.polyfit(ethiopia_df['Unemployment rate, (aged 15 over) (%)'].values, ethiopia_df['Inflation, consumer prices (annual %)'].values, 1)

plt.plot(ethiopia_df['Unemployment rate, (aged 15 over) (%)'].values, m*ethiopia_df['Unemployment rate, (aged 15 over) (%)'].values+b, color='red', linestyle='-', label='Regression Line')

plt.legend()

# Display the plot using Streamlit
st.pyplot(fig)

st.write("""| Parameter      | Value |
| ----------- | ----------- |
| MSE      | 1.4443547605931981e+20 |
| R2   | 0.9031126967632856 |""")