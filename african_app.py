import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from functions import clean_dataset

# Load the data
african_economies_df = pd.read_csv('african_economies_dataset.csv')


st.title("African Economies Analysis")
st.subheader("African Economies Dataset")
st.write("I found the dataset in the African Development Bank Group (ADBG) link to the website: https://dataportal.opendataforafrica.org/nbyenxf/afdb-socio-economic-database-1960-2022")

# I create a sidebar with all the controls
st.sidebar.header('Controls')
st.sidebar.write('Select what do you want to see:')
if st.sidebar.checkbox('EDA'):
    st.header('Exaploratory Data Analysis')

    # Displaying the columns of the dataframe
    st.markdown("""
##### Dataframe Data:
    Annual growth rate of real GDP per capita (%)                            
    CO2 emissions (metric tons per capita)                                  
    Capital and financial account balance (Net, BoP, cur. US$)               
    Central government, Fiscal Balance (Current US $)                       
    Central government, total expenditure and net lending  (Current US $)   
    Consumer Price Index, Total (Annual Growth rates, %)                     
    Consumer Price Index, in Energy (Annual Growth rates, %)                 
    Consumer Price Index, in Food (Annual Growth rates, %)                   
    Exports of goods & services, Value (WEO, cur. US$)                       
    GDP (current US$)                                                        
    GDP per capita,(current US$)                                             
    General government final consumption expenditure (current US$)           
    Gross capital formation (current US$)                                    
    Gross capital formation, Private sector  (current US$)                   
    Gross capital formation, Public sector  (current US$)                    
    Health expenditure per capita (current US$)                              
    Household final consumption expenditure (current US$)                    
    Imports of goods & services, Value (WEO, cur. US$)                       
    Inflation, consumer prices (annual %)                                    
    Manufacturing, value added (current US$)                                 
    Oil imports, Value (Cur. USD)                                            
    Population, total                                                        
    Public Administration and defence, value added (current US$)             
    Real GDP growth (annual %)                                               
    Trade balance (Net, BoP, cur. US$)                                       
    Unemployment rate, (aged 15 over) (%)
"""
)
    # Creating 2 checkbox to show the raw dataset or the cleaned one
    st.write('Choose to see the EDA of the raw dataset or after its cleaning:')
    if st.checkbox('Raw Dataset'):
        st.write('African Economies dataset:')

        # Overview of the all dataset and the shape
        st.write(african_economies_df)
        st.write('Rows and Columns:', african_economies_df.shape)
        
        # Showing the head and tail of the dataset
        st.write('Dataset Head and Tail:')
        st.write(african_economies_df.head())
        st.write(african_economies_df.tail())

        # Showing some statistical informations
        st.write('Some Statistical Informations:')
        st.write(african_economies_df.describe().T)

    # Cleaning the dataset with the function clean_dataset and displaying it
    african_economies_clean_df = clean_dataset()
    
    # Show the clean dataset
    if st.checkbox('Clean Dataset'):
        st.write('Below is shown the clean dataset, to see into more detail the cleaning procedure check the notebook.')
        
         # Overview of the all dataset and the shape
        st.write('African Economies Dataset:')
        st.write(african_economies_clean_df)
        st.write('Rows and Columns:', african_economies_clean_df.shape)

        # Showing the head and tail of the dataset
        st.write('Dataset Head and Tail:')
        st.write(african_economies_clean_df.head())
        st.write(african_economies_clean_df.tail())

        # Showing some statistical informations
        st.write('Some Statistical Informations:')
        st.write(african_economies_clean_df.describe())

# Using the clean dataset and creating a groupby dataset
african_economies_clean_df = clean_dataset()
african_groupby_mean_df = african_economies_clean_df.groupby('country').mean()

# The checkbox that shows the plots
if st.sidebar.checkbox('Plots'):
    st.header('Plots')

    # I create a selectbox to allow the user to show the plot he/she chooses
    plot_select = st.selectbox('Select a plot to show:', ['Average GDP Growth', 'Average GDP per capita Growth', 'Population', 'Inflation Trend', 'CO2 Emissions per Capita'], key = 'plot_key')
        
    # Using a if statement for the box selected to show the corresponding graph
    if plot_select == 'Average GDP Growth':

        # Sorting the grouped dataset by Real GDP growth (annual %)
        real_gdp_growth_sorted = african_groupby_mean_df.sort_values(by='Real GDP growth (annual %)',ascending=False)
        
        # Set the size of the figure
        fig1 = plt.figure(figsize=(15,10))

        # Barplot of the real GDP per capita growth
        plt.bar(x=real_gdp_growth_sorted.index, height=real_gdp_growth_sorted['Real GDP growth (annual %)'])
        plt.xticks(rotation=90)
        plt.title('Real GDP growth (annual %)')
        plt.ylabel('Growth Rate (%)')
        plt.xlabel('Country')
        plt.show()
        st.pyplot(fig1)

        # Creating another selectbox that gives the user the posibility to see into more detail the trend of the first 5 countries
        country_select = st.selectbox('Select a country to show the trend in detail:',real_gdp_growth_sorted.index[:5], key = 'country_key')
        
        # If the fist country is selected
        if country_select == real_gdp_growth_sorted.index[0]:

            # I create a boolean mask and assign the first country
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_growth_sorted.index[0]
            country_df = african_economies_clean_df[country_mask]
            
            # Plot the GDP growth trends for first country
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], color='blue', label='Real GDP growth (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Real GDP growth (annual %)')
            ax[0].set_title(f'Line Plot: GDP Growth Trends of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], alpha=1, color='red', label='Real GDP growth (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Real GDP growth (annual %)')
            ax[1].set_title(f'Bar Plot: GDP Growth Trends of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == real_gdp_growth_sorted.index[1]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_growth_sorted.index[1]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], color='blue', label='Real GDP growth (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Real GDP growth (annual %)')
            ax[0].set_title(f'Line Plot: GDP Growth Trends of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], alpha=1, color='red', label='Real GDP growth (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Real GDP growth (annual %)')
            ax[1].set_title(f'Bar Plot: GDP Growth Trends of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == real_gdp_growth_sorted.index[2]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_growth_sorted.index[2]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], color='blue', label='Real GDP growth (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Real GDP growth (annual %)')
            ax[0].set_title(f'Line Plot: GDP Growth Trends of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], alpha=1, color='red', label='Real GDP growth (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Real GDP growth (annual %)')
            ax[1].set_title(f'Bar Plot: GDP Growth Trends of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == real_gdp_growth_sorted.index[3]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_growth_sorted.index[3]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], color='blue', label='Real GDP growth (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Real GDP growth (annual %)')
            ax[0].set_title(f'Line Plot: GDP Growth Trends of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], alpha=1, color='red', label='Real GDP growth (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Real GDP growth (annual %)')
            ax[1].set_title(f'Bar Plot: GDP Growth Trends of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == real_gdp_growth_sorted.index[4]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_growth_sorted.index[4]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], color='blue', label='Real GDP growth (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Real GDP growth (annual %)')
            ax[0].set_title(f'Line Plot: GDP Growth Trends of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], alpha=1, color='red', label='Real GDP growth (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Real GDP growth (annual %)')
            ax[1].set_title(f'Bar Plot: GDP Growth Trends of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

    if plot_select == 'Average GDP per capita Growth':

        real_gdp_per_capita_growth_sorted = african_groupby_mean_df.sort_values(by='Annual growth rate of real GDP per capita (%)',ascending=False)
        
        # Set the size of the figure
        fig1 = plt.figure(figsize=(15,10))

        # Barplot of the real GDP per capita growth
        plt.bar(x=real_gdp_per_capita_growth_sorted.index, height=real_gdp_per_capita_growth_sorted['Annual growth rate of real GDP per capita (%)'])
        plt.xticks(rotation=90)
        plt.title('Average GDP per capita Growth rate')
        plt.ylabel('Growth Rate (%)')
        plt.xlabel('Country')
        plt.show()
        st.pyplot(fig1)

        country_select = st.selectbox('Select a country to:',real_gdp_per_capita_growth_sorted.index[:5], key = 'country_key2')
        
        if country_select == real_gdp_per_capita_growth_sorted.index[0]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_per_capita_growth_sorted.index[0]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Annual growth rate of real GDP per capita (%)'], color='blue', label='Annual growth rate of real GDP per capita (%)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[0].set_title(f'Line Plot: GDP per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Annual growth rate of real GDP per capita (%)'], alpha=1, color='red', label='Annual growth rate of real GDP per capita (%)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[1].set_title(f'Bar Plot: GDP per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == real_gdp_per_capita_growth_sorted.index[1]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_per_capita_growth_sorted.index[1]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Annual growth rate of real GDP per capita (%)'], color='blue', label='Annual growth rate of real GDP per capita (%)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[0].set_title(f'Line Plot: GDP per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], alpha=1, color='red', label='Annual growth rate of real GDP per capita (%)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[1].set_title(f'Bar Plot: GDP per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == real_gdp_per_capita_growth_sorted.index[2]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_per_capita_growth_sorted.index[2]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Annual growth rate of real GDP per capita (%)'], color='blue', label='Annual growth rate of real GDP per capita (%)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[0].set_title(f'Line Plot: GDP per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Annual growth rate of real GDP per capita (%)'], alpha=1, color='red', label='Annual growth rate of real GDP per capita (%)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[1].set_title(f'Bar Plot: GDP per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == real_gdp_per_capita_growth_sorted.index[3]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_per_capita_growth_sorted.index[3]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Annual growth rate of real GDP per capita (%)'], color='blue', label='Real GDP growth (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[0].set_title(f'Line Plot: GDP per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Annual growth rate of real GDP per capita (%)'], alpha=1, color='red', label='Annual growth rate of real GDP per capita (%)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[1].set_title(f'Bar Plot: GDP per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == real_gdp_per_capita_growth_sorted.index[4]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == real_gdp_per_capita_growth_sorted.index[4]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Real GDP growth (annual %)'], color='blue', label='Annual growth rate of real GDP per capita (%)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[0].set_title(f'Line Plot: GDP per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Annual growth rate of real GDP per capita (%)'], alpha=1, color='red', label='Annual growth rate of real GDP per capita (%)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Annual growth rate of real GDP per capita (%)')
            ax[1].set_title(f'Bar Plot: GDP per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

    if plot_select == 'Population':
        # Creating two mask to compare how has changed the distribution of Population
        mask_1995 = african_economies_clean_df.index.get_level_values('Date') == 1995
        mask_2023 = african_economies_clean_df.index.get_level_values('Date') == 2023
        african_economies_1995_df = african_economies_clean_df[mask_1995]
        african_economies_2023_df = african_economies_clean_df[mask_2023]

        population_sorted = african_economies_2023_df.sort_values(by='Population, total',ascending=False)

        # Create boxplots for population distribution in 1995 and 2023
        fig1 = plt.figure(figsize=(15,10))
        plt.boxplot(x=[african_economies_1995_df['Population, total'], african_economies_2023_df['Population, total']],
                    vert=True,
                    patch_artist=True,
                    boxprops=  dict(facecolor = "lightblue"),
                    medianprops= dict(color = "red"), 
                    labels=['1995','2023'])
        # Set the title and labels of the plot
        plt.title('Population Distribution in 1995 and 2023')
        plt.ylabel('Population')
        plt.show()
        st.pyplot(fig1)

        country_select = st.selectbox('Select a plot to show:',population_sorted.index.get_level_values('country')[:5], key = 'country_key3')
        
        if country_select == population_sorted.index.get_level_values('country')[0]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == population_sorted.index.get_level_values('country')[0]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Population, total'], color='blue', label='Population, total', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Population')
            ax[0].set_title(f'Line Plot: Population Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Population, total'], alpha=1, color='red', label='Population, total')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Population')
            ax[1].set_title(f'Bar Plot: Population Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == population_sorted.index.get_level_values('country')[1]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == population_sorted.index.get_level_values('country')[1]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Population, total'], color='blue', label='Population, total', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Population')
            ax[0].set_title(f'Line Plot: Population Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Population, total'], alpha=1, color='red', label='Population, total')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Population')
            ax[1].set_title(f'Bar Plot: Population Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == population_sorted.index.get_level_values('country')[2]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == population_sorted.index.get_level_values('country')[2]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Population, total'], color='blue', label='Population, total', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Population')
            ax[0].set_title(f'Line Plot: Population Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Population, total'], alpha=1, color='red', label='Population, total')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Population')
            ax[1].set_title(f'Bar Plot: Population Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == population_sorted.index.get_level_values('country')[3]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == population_sorted.index.get_level_values('country')[3]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Population, total'], color='blue', label='Population, total', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Population')
            ax[0].set_title(f'Line Plot: Population Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Population, total'], alpha=1, color='red', label='Population, total')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Population')
            ax[1].set_title(f'Bar Plot: Population Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == population_sorted.index.get_level_values('country')[4]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == population_sorted.index.get_level_values('country')[4]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Population, total'], color='blue', label='Population, total', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Population')
            ax[0].set_title(f'Line Plot: Population Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Population, total'], alpha=1, color='red', label='Population, total')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Population')
            ax[1].set_title(f'Bar Plot: Population Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

    if plot_select == 'Inflation Trend':

        inflation_sorted = african_groupby_mean_df.sort_values(by='Inflation, consumer prices (annual %)',ascending=False)
        # Set the size of the figure
        
        fig1 = plt.figure(figsize=(15,10))

        # Barplot of the real GDP per capita growth
        plt.bar(x=inflation_sorted.index, height=inflation_sorted['Inflation, consumer prices (annual %)'])
        plt.xticks(rotation=90)
        plt.title('Inflation, consumer prices (annual %)')
        plt.ylabel('Growth Rate (%)')
        plt.xlabel('Country')
        plt.show()
        st.pyplot(fig1)

        country_select = st.selectbox('Select a plot to show:', inflation_sorted.index[:5], key = 'country_key4')
        
        if country_select == inflation_sorted.index[0]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == inflation_sorted.index[0]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], color='blue', label='Inflation, consumer prices (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Inflation, consumer prices (annual %)')
            ax[0].set_title(f'Line Plot: Inflation Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], alpha=1, color='red', label='Inflation, consumer prices (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Inflation, consumer prices (annual %)')
            ax[1].set_title(f'Bar Plot: Inflation Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == inflation_sorted.index[1]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == inflation_sorted.index[1]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], color='blue', label='Inflation, consumer prices (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Inflation, consumer prices (annual %)')
            ax[0].set_title(f'Line Plot: Inflation Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], alpha=1, color='red', label='Inflation, consumer prices (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Inflation, consumer prices (annual %)')
            ax[1].set_title(f'Bar Plot: Inflation Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == inflation_sorted.index[2]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == inflation_sorted.index[2]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], color='blue', label='Inflation, consumer prices (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Inflation, consumer prices (annual %)')
            ax[0].set_title(f'Line Plot: Inflation Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], alpha=1, color='red', label='Inflation, consumer prices (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Inflation, consumer prices (annual %)')
            ax[1].set_title(f'Bar Plot: Inflation Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == inflation_sorted.index[3]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == inflation_sorted.index[3]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], color='blue', label='Inflation, consumer prices (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Inflation, consumer prices (annual %)')
            ax[0].set_title(f'Line Plot: Inflation Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], alpha=1, color='red', label='Inflation, consumer prices (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Inflation, consumer prices (annual %)')
            ax[1].set_title(f'Bar Plot: Inflation Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)


        if country_select == inflation_sorted.index[4]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == inflation_sorted.index[4]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], color='blue', label='Inflation, consumer prices (annual %)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('Inflation, consumer prices (annual %)')
            ax[0].set_title(f'Line Plot: Inflation Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['Inflation, consumer prices (annual %)'], alpha=1, color='red', label='Inflation, consumer prices (annual %)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Inflation, consumer prices (annual %)')
            ax[1].set_title(f'Bar Plot: Inflation Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

    if plot_select == 'CO2 Emissions per Capita':

        co2_sorted = african_groupby_mean_df.sort_values(by='CO2 emissions (metric tons per capita)',ascending=False)
        # Set the size of the figure
        
        fig1 = plt.figure(figsize=(15,10))

        # Barplot of the real GDP per capita growth
        plt.bar(x=co2_sorted.index, height=co2_sorted['CO2 emissions (metric tons per capita)'])
        plt.xticks(rotation=90)
        plt.title('CO2 emissions (metric tons per capita)')
        plt.ylabel('Growth Rate (%)')
        plt.xlabel('Country')
        plt.show()
        st.pyplot(fig1)

        country_select = st.selectbox('Select a plot to show:',co2_sorted.index[:5], key = 'country_key4')
        
        if country_select == co2_sorted.index[0]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == co2_sorted.index[0]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], color='blue', label='CO2 emissions (metric tons per capita)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[0].set_title(f'Line Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], alpha=1, color='red', label='CO2 emissions (metric tons per capita)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[1].set_title(f'Bar Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == co2_sorted.index[1]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == co2_sorted.index[1]
            country_df = african_economies_clean_df[country_mask]
            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], color='blue', label='CO2 emissions (metric tons per capita)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[0].set_title(f'Line Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], alpha=1, color='red', label='CO2 emissions (metric tons per capita)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[1].set_title(f'Bar Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == co2_sorted.index[2]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == co2_sorted.index[2]
            country_df = african_economies_clean_df[country_mask]

            # Plot the GDP growth trends for Ethiopia
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], color='blue', label='CO2 emissions (metric tons per capita)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[0].set_title(f'Line Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], alpha=1, color='red', label='CO2 emissions (metric tons per capita)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[1].set_title(f'Bar Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == co2_sorted.index[3]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == co2_sorted.index[3]
            country_df = african_economies_clean_df[country_mask]
            
            # Plot the GDP growth trends for fourth country
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], color='blue', label='CO2 emissions (metric tons per capita)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[0].set_title(f'Line Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], alpha=1, color='red', label='CO2 emissions (metric tons per capita)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[1].set_title(f'Bar Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

        if country_select == co2_sorted.index[4]:
            country_mask = african_economies_clean_df.index.get_level_values('country') == co2_sorted.index[4]
            country_df = african_economies_clean_df[country_mask]
            
            # Plot the GDP growth trends for fifth country
            fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
            # Subplot 1: Line plot
            ax[0].plot(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], color='blue', label='CO2 emissions (metric tons per capita)', linestyle='-', marker='o')
            ax[0].set_xlabel('Time (year)')
            ax[0].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[0].set_title(f'Line Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[0].grid(linestyle='--', alpha=0.6)
            ax[0].set_xticks(country_df.index.get_level_values('Date'))
            ax[0].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[0].legend()

            # Subplot 2: Bar chart
            ax[1].bar(country_df.index.get_level_values('Date'), country_df['CO2 emissions (metric tons per capita)'], alpha=1, color='red', label='CO2 emissions (metric tons per capita)')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('CO2 emissions (metric tons per capita)')
            ax[1].set_title(f'Bar Plot: CO2 emissions per Capita Trend of {country_select}')
            ax[1].grid(linestyle='--', alpha=0.6)
            ax[1].set_xticks(country_df.index.get_level_values('Date'))
            ax[1].set_xticklabels(country_df.index.get_level_values('Date'), rotation=45)
            ax[1].legend()
            plt.tight_layout()  # Ensure proper spacing between subplots
            plt.show()
            st.pyplot(fig2)

    
    african_economies_clean_df = clean_dataset()
    
    # Creating a checkbox that it is shown when the plot checkbox is ticked
    if st.sidebar.checkbox('Correlations'):
    
        st.header('Correlations')

        # A selectbox to show the correlation matrix of a specific country
        corr_select = st.selectbox('Select a country to show the Correlation Matrix:',african_economies_clean_df.index.get_level_values('country').unique(), key = 'corr_key')

        # If statement to plot the correlation matrix
        if corr_select:

            # Boolean mask to select country selected by the user
            corr_country_mask = african_economies_clean_df.index.get_level_values('country') == corr_select
            corr_country_df = african_economies_clean_df[corr_country_mask]
            
            # Plotting the correlation matrix
            fig_heat = plt.figure(figsize=(20,20))
            sb.heatmap(data=corr_country_df.corr(),annot=True, cmap='coolwarm')
            plt.show()
            st.pyplot(fig_heat)

            st.subheader('Plotting some correlations')

            # Selection for the features to be show in specific
            feature_corr = st.multiselect('Select **two** features', corr_country_df.columns, max_selections=2, key='my_key')

            if len(feature_corr) == 2:
                with st.spinner('Plotting...'):
                    
                    # Plotting the correlation between the two selected features
                    fig_corr, ax = plt.subplots(figsize=(10, 6))

                    plt.scatter(corr_country_df[feature_corr[0]],corr_country_df[feature_corr[1]], color='blue',label=feature_corr[0], marker='o')

                    plt.xlabel(feature_corr[0])
                    plt.ylabel(feature_corr[1])
                    plt.title(f'{feature_corr[0]} vs {feature_corr[1]}')
                    plt.grid(True)

                    # Creating the regression line
                    m, b = np.polyfit(corr_country_df[feature_corr[0]].values, corr_country_df[feature_corr[1]].values, 1)

                    # Plotting the regression line
                    plt.plot(corr_country_df[feature_corr[0]].values, m*corr_country_df[feature_corr[0]].values+b, color='red', linestyle='-', label='Regression Line')

                    plt.legend()

                    # Display the plot
                    st.pyplot(fig_corr)

african_economies_clean_df = clean_dataset()
ethiopia_mask = african_economies_clean_df.index.get_level_values('country') == 'Ethiopia'
ethiopia_df = african_economies_clean_df[ethiopia_mask]

# The checkbox shows the section of the model when clicked
if st.sidebar.checkbox('Model'):
    st.header('A model to predict the GDP')
    ethiopia_model_df = ethiopia_df.reset_index().drop(['country'],axis=1).set_index('Date')
    
    # The container is used to include all the following
    container = st.container()

    # I create a checkbox that allows to select all the features
    all_features = st.checkbox('Select all')

    # Create a if statement that when the all_features is Trues it selects all the features
    if all_features:
        
        # A multiselect that allows to select the features to be used in the model
        selected_features = container.multiselect('Select one or more features:', ['Central government, total expenditure and net lending  (Current US $)',
       'Consumer Price Index, in Food (Annual Growth rates, %)',
       'Gross capital formation (current US$)',
       'Health expenditure per capita (current US$)',
       'Household final consumption expenditure (current US$)',
       'Inflation, consumer prices (annual %)',
       'Population, total',
       'Public Administration and defence, value added (current US$)'],
       ['Central government, total expenditure and net lending  (Current US $)',
       'Consumer Price Index, in Food (Annual Growth rates, %)',
       'Gross capital formation (current US$)',
       'Health expenditure per capita (current US$)',
       'Household final consumption expenditure (current US$)',
       'Inflation, consumer prices (annual %)',
       'Population, total',
       'Public Administration and defence, value added (current US$)'])
    else:
        selected_features = container.multiselect('Select one or more features:', ['Central government, total expenditure and net lending  (Current US $)',
       'Consumer Price Index, in Food (Annual Growth rates, %)',
       'Gross capital formation (current US$)',
       'Health expenditure per capita (current US$)',
       'Household final consumption expenditure (current US$)',
       'Inflation, consumer prices (annual %)',
       'Population, total',
       'Public Administration and defence, value added (current US$)'])

    # I create a slider that allows to change the training size
    train_size = st.slider("Select the training size:", min_value=0.1, max_value=0.9, step=0.1)

    # I assign the data that I want to predict
    if len(selected_features) > 0 and st.button('RUN MODEL'):
        with st.spinner('Training...'):

            # Assigning the variable, X changes based on the user selection of the features
            y = ethiopia_model_df['GDP (current US$)']  # response
            X = ethiopia_model_df[selected_features] # predictors

            # Splitting the data into train and test based on the user selection
            X_train, y_train =X[:int(len(X)*train_size)], y[:int(len(X)*train_size)]
            X_test, y_test = X[int(len(X)*train_size):], y[int(len(X)*train_size):]

            # Fitting the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Creating a new dataframe containing the predictions
            y_pred = pd.DataFrame(index=y_test.index)
            y_pred['predicted_gdp'] = model.predict(X_test)

            # Plotting the model
            fig_model = plt.figure(figsize=(10,6))
            plt.plot(y_pred['predicted_gdp'], '--', label='Predicted GDP')
            plt.plot(y_test, '-', label='Test Data')

            plt.xlabel('Year')
            plt.ylabel('GDP (current USD)')
            plt.title('Predicted GDP vs Test Data')
            plt.xticks(y_test.index.get_level_values(0)[::2], rotation = 45)
            plt.legend()

            plt.show()
            st.pyplot(fig_model)

            # Calculate the MSE and R2
            mse = mean_squared_error(y_test, y_pred['predicted_gdp'])
            r2 = r2_score(y_test, y_pred['predicted_gdp'])

            # Print out the MSE and R2
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Coefficient of Determination (R2): {r2:.2f}')