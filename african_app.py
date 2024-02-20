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
# Display the average increase of GDP per capita

st.title("African Economies Analysis")
st.subheader("African Economies Dataset")
st.write("I found the dataset in the African Development Bank Group (ADBG) link to the website: https://dataportal.opendataforafrica.org/nbyenxf/afdb-socio-economic-database-1960-2022")
st.sidebar.header('Controls')
st.sidebar.write('Select what do you want to see:')
if st.sidebar.checkbox('EDA'):
    st.header('Exaploratory Data Analysis')
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
    st.write('Choose to see the EDA of the dataframe before or after its cleaning:')
    if st.checkbox('Raw Dataset'):
        st.write('African Economies dataframe:')
        st.write(african_economies_df)
        st.write('Rows and columns:', african_economies_df.shape)
        
        st.write('Dataframe head and tail:')
        st.write(african_economies_df.head())
        st.write(african_economies_df.tail())

        st.write('Some numerical informations:')
        st.write(african_economies_df.describe().T)

    african_economies_clean_df = clean_dataset()
    if st.checkbox('Clean Dataset'):
        st.write('Below is shown the clean dataset, to see into more detail the cleaning procedure check the notebook.')
        st.write('Pokemon dataframe:')
        st.write(african_economies_clean_df)
        st.write('Rows and columns:', african_economies_clean_df.shape)

        st.write('Dataframe head and tail:')
        st.write(african_economies_clean_df.head())
        st.write(african_economies_clean_df.tail())

        st.write('Some numerical informations:')
        st.write(african_economies_clean_df.describe())

african_economies_clean_df = clean_dataset()
if st.sidebar.checkbox('Plots'):
    st.header('Plots')

    african_groupby_mean_df = african_economies_clean_df.groupby('country').mean()

    # Sort the dataframe by GDP per capita and GDP growth
    real_gdp_per_capita_growth_sorted = african_groupby_mean_df.sort_values(by='Annual growth rate of real GDP per capita (%)',ascending=False)
    real_gdp_growth_sorted = african_groupby_mean_df.sort_values(by='Real GDP growth (annual %)',ascending=False)

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

    ethiopia_mask = african_economies_clean_df.index.get_level_values('country') == 'Ethiopia'
    ethiopia_df = african_economies_clean_df[ethiopia_mask]

    # Plot the GDP growth trends for Ethiopia
    fig2, ax = plt.subplots(2, 1, figsize=(12, 10))
    # Subplot 1: Line plot
    ax[0].plot(ethiopia_df.index.get_level_values('Date'), ethiopia_df['Real GDP growth (annual %)'], color='blue', label='Real GDP growth (annual %)', linestyle='-', marker='o')
    ax[0].set_xlabel('Time (year)')
    ax[0].set_ylabel('Real GDP growth (annual %)')
    ax[0].set_title('Line Plot: Ethiopian GDP Growth Trends')
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

    st.header('Correlations')

    fig_heat = plt.figure(figsize=(20,20))
    sb.heatmap(data=ethiopia_df.corr(),annot=True)
    plt.show()

    st.pyplot(fig_heat)

    st.subheader('Plotting some correlations')
    feature_corr = st.multiselect('Select **two** features', ethiopia_df.columns, max_selections=2, key='my_key')

    if len(feature_corr) == 2:
        with st.spinner('Plotting...'):
        # Plotting the Phillips curve
            fig_corr, ax = plt.subplots(figsize=(10, 6))

            plt.scatter(ethiopia_df[feature_corr[0]],ethiopia_df[feature_corr[1]], color='blue',label=feature_corr[0], marker='o')

            plt.xlabel(feature_corr[0])
            plt.ylabel(feature_corr[1])
            plt.title(f'{feature_corr[0]} vs {feature_corr[1]}')
            plt.grid(True)

            m, b = np.polyfit(ethiopia_df[feature_corr[0]].values, ethiopia_df[feature_corr[1]].values, 1)

            plt.plot(ethiopia_df[feature_corr[0]].values, m*ethiopia_df[feature_corr[0]].values+b, color='red', linestyle='-', label='Regression Line')

            plt.legend()

            # Display the plot
            st.pyplot(fig_corr)

african_economies_clean_df = clean_dataset()
ethiopia_mask = african_economies_clean_df.index.get_level_values('country') == 'Ethiopia'
ethiopia_df = african_economies_clean_df[ethiopia_mask]

if st.sidebar.checkbox('Model'):
    st.header('A model to predict the GDP')
    ethiopia_model_df = ethiopia_df.reset_index().drop(['country'],axis=1).set_index('Date')

    y = ethiopia_model_df['GDP (current US$)'] # depent variable
    
    container = st.container()

    all_features = st.checkbox('Select all')

    if all_features:
        selected_features = container.multiselect('Select one or more features:',['Annual growth rate of real GDP per capita (%)',
       'CO2 emissions (metric tons per capita)',
       'Capital and financial account balance (Net, BoP, cur. US$)',
       'Central government, Fiscal Balance (Current US $)',
       'Central government, total expenditure and net lending  (Current US $)',
       'Consumer Price Index, Total (Annual Growth rates, %)',
       'Consumer Price Index, in Energy (Annual Growth rates, %)',
       'Consumer Price Index, in Food (Annual Growth rates, %)',
       'Exports of goods & services, Value (WEO, cur. US$)',
       'GDP per capita,(current US$)',
       'General government final consumption expenditure (current US$)',
       'Gross capital formation (current US$)',
       'Gross capital formation, Private sector  (current US$)',
       'Gross capital formation, Public sector  (current US$)',
       'Health expenditure per capita (current US$)',
       'Household final consumption expenditure (current US$)',
       'Imports of goods & services, Value (WEO, cur. US$)',
       'Inflation, consumer prices (annual %)',
       'Manufacturing, value added (current US$)',
       'Oil imports, Value (Cur. USD)', 'Population, total',
       'Public Administration and defence, value added (current US$)',
       'Real GDP growth (annual %)', 'Trade balance (Net, BoP, cur. US$)',
       'Unemployment rate, (aged 15 over) (%)'],
       ['Annual growth rate of real GDP per capita (%)',
       'CO2 emissions (metric tons per capita)',
       'Capital and financial account balance (Net, BoP, cur. US$)',
       'Central government, Fiscal Balance (Current US $)',
       'Central government, total expenditure and net lending  (Current US $)',
       'Consumer Price Index, Total (Annual Growth rates, %)',
       'Consumer Price Index, in Energy (Annual Growth rates, %)',
       'Consumer Price Index, in Food (Annual Growth rates, %)',
       'Exports of goods & services, Value (WEO, cur. US$)',
       'GDP per capita,(current US$)',
       'General government final consumption expenditure (current US$)',
       'Gross capital formation (current US$)',
       'Gross capital formation, Private sector  (current US$)',
       'Gross capital formation, Public sector  (current US$)',
       'Health expenditure per capita (current US$)',
       'Household final consumption expenditure (current US$)',
       'Imports of goods & services, Value (WEO, cur. US$)',
       'Inflation, consumer prices (annual %)',
       'Manufacturing, value added (current US$)',
       'Oil imports, Value (Cur. USD)', 'Population, total',
       'Public Administration and defence, value added (current US$)',
       'Real GDP growth (annual %)', 'Trade balance (Net, BoP, cur. US$)',
       'Unemployment rate, (aged 15 over) (%)'])
    else:
        selected_features = container.multiselect('Select one or more features:', ethiopia_model_df.columns.difference(['GDP (current US$)']))

    test_size = st.slider("Select the training size:", min_value=0.1, max_value=0.9, step=0.1)

    # I assign the data that I want to predict
    if len(selected_features) > 0 and st.button('RUN MODEL'):
        with st.spinner('Training...'):
            X = ethiopia_model_df[selected_features]  # predictors
            X_train, y_train =X[:int(len(X)*test_size)], y[:int(len(X)*test_size)]
            X_test, y_test = X[int(len(X)*test_size):], y[int(len(X)*test_size):]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = pd.DataFrame(index=y_test.index)
            y_pred['predicted_gdp'] = model.predict(X_test)

            fig_model = plt.figure(figsize=(10,6))
            plt.plot(y_pred['predicted_gdp'], '--', label='Predicted GDP')
            plt.plot(y_test, '-', label='Test Data')

            plt.xlabel('Year')
            plt.ylabel('GDP (current USD)')
            plt.title('Predicted GDP vs Test Data')
            plt.xticks(y_test.index, rotation = 45)
            plt.legend()

            plt.show()
            st.pyplot(fig_model)

            mse = mean_squared_error(y_test, y_pred['predicted_gdp'])
            r2 = r2_score(y_test, y_pred['predicted_gdp'])

            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Coefficient of Determination (R2): {r2:.2f}')