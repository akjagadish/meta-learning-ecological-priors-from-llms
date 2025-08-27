import pandas as pd

# converted .dat to .csv by pasting dat contents into a .csv file and adding column names

# load data
stimuli = pd.read_csv('./Little2022_stimuli.csv')
choices = pd.read_csv('./Little2022_choices.csv')

# add new column to stimuli
stimuli['type'] = 'train'

# add new column to choices
choices['type'] = 'test'

# # Drop the Unnamed: 0 column since it is just an index
# choices.drop(columns=['Unnamed: 0'], inplace=True)
# stimuli.drop(columns=['Unnamed: 0'], inplace=True)

# Merge choices_df with the relevant columns from stimuli_df
merged_df = choices.merge(stimuli[['participant', 'task', 'num_points', 'scale', 'source']],
                             on=['participant', 'task'],
                             how='left')

# Display the first few rows of the merged dataframe to verify the result
merged_df.head()

# combine stimuli and choices
combined = pd.concat([stimuli, merged_df])

# save data
combined.to_csv('./little2022functionestimation.csv')