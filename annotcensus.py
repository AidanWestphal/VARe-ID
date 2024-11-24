import pandas as pd

# Load the CSV file
df = pd.read_csv('test_dataset/post_your_file.csv')  # replace 'your_file.csv' with the actual file path

# Create the "annot census" column based on the condition for "viewpoint"
# df['annot census'] = df['viewpoint'].apply(lambda x: True if x == 'Right' else False)

# Add the "image fname" column by appending ".jpg" to the "image uuid" column
df['image fname'] = df['image uuid'] + '.jpg'

# # Display the modified DataFrame
# print(df.head())
output_filename = 'posttt_' + 'your_file.csv'  # adjust this if necessary
df.to_csv(output_filename, index=False)

print(f"Modified file saved as {output_filename}")


