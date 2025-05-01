import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = 'Cancer_Data.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the splits to new CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

small_train_df, small_test_df = train_test_split(test_df, test_size=0.2, random_state=42)

small_train_df.to_csv('small_train.csv', index=False)
small_test_df.to_csv('small_test.csv', index=False)
print("Data split complete: 'train.csv' and 'test.csv' created.")