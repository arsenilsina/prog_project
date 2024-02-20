import pandas as pd
def clean_dataset():
    african_economies_df = pd.read_csv('african_economies_dataset.csv')
    # Now I work with the clean dataset -> no null values
    # Create a multiindex setting the country as index and using the 'indicators' as columns
    african_economies_new_df = african_economies_df.pivot_table(values='Value', index=('country','Date'), columns='indicator')
    # Filter the dataframe to only include data from 1995 onwards
    date_mask = african_economies_new_df.index.get_level_values('Date') > 1994
    african_economies_21st_df = african_economies_new_df.loc[date_mask]
    # Create an interpolation line to fill the NaN values
    african_economies_21st_clean_df = african_economies_21st_df.apply(lambda group: group.infer_objects(copy=False).interpolate(method='linear'))
    # Replace the 0 values with NaN
    for col in african_economies_21st_clean_df.columns:
        african_economies_21st_clean_df[col] = african_economies_21st_clean_df[col].replace(0, pd.NA)
    #Drop the columns that presents NaN
    african_economies_21st_clean_df = african_economies_21st_clean_df.drop(columns=['Oil exports, Value (Cur. USD)', 'External Public Debt from Private Creditors (Current USD)'])
    # I create a boolean mask with the indexes to be dropped
    drop_index_mask = african_economies_21st_clean_df.index.get_level_values(0).isin(['Algeria', 'Burundi', 'Central African Republic', 'Eritrea', 'Gambia',
        'Seychelles', 'Somalia', 'South Sudan'])
    # I create a new dataframe wich contains the indexes to be dropped
    drop_df = african_economies_21st_clean_df[drop_index_mask]
    african_economies_21st_clean_df.drop(drop_df.index, inplace=True)
    # I convert the values all into float type
    african_economies_21st_clean_df = african_economies_21st_clean_df.astype(float)
    return african_economies_21st_clean_df