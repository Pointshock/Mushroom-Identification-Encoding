# Import libraries
from turtle import pd
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

# Define a data file to import
mushroom_data_file = "mushrooms.csv"
# Import our data
df = pd.read_csv(mushroom_data_file, encoding='ISO-8859-1', engine='python')

# Feature selection
pre_feature_list = df[['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']]

# Print out the data types of each feature
print("\n\t *** Feature data types ***\n")
print(pre_feature_list.dtypes)
print("_" * 50)

input("Press Enter to Continue...")

# Print out the first and last 5 rows of the feature list
print("\n\t *** Feature List ***\n")
print(pre_feature_list.head())
print(pre_feature_list.tail())
print("_" * 50)
# Print out the full data shape
print("\nFull data shape: " , pre_feature_list.shape)
print("_" * 50)

input("Press Enter to Continue...")

# Print out the unique values for each column feature
print("\n\t *** Unique values for each column feature ***\n")
print(pre_feature_list.nunique())
print("\n\n\n")
# Print out the unique values for each column feature
print("\n\t *** Unique values for each column feature ***\n")
for col in pre_feature_list.columns:
    print(col,": ", pre_feature_list[col].unique())
print("_" * 50)

input("Press Enter to Continue...")

# Now we need to convert the categorical data into numerical data
# We will use the OneHotEncoder to do this

# Create an instance of the OneHotEncoder
enc = OneHotEncoder(sparse=False)

# Fit and transform the encoder to the feature list
encoded_feature_array = enc.fit_transform(pre_feature_list)

# Print out the encoded feature list
print("\n\t *** Encoded feature list ***\n")
print(encoded_feature_array)
print("_" * 50)

input("Press Enter to Continue...")

# Print out the encoded feature list categories
print("\n\t *** Encoded feature list categories ***\n")
print(enc.categories_)
print("_" * 50)

# Now it is time for label/ target selection
pre_label_list = df[['class']]

# Print out the unique values for each column target
print("\n\t *** Unique values for each column target ***\n")
for col in pre_label_list.columns:
    print(col,": ", pre_label_list[col].unique())

print("_" * 50)

# Now we will encode the target list
encoded_target_array = enc.fit_transform(pre_label_list)

input("Press Enter to Continue...")

print("\n\n\n\t *** This is the target/ label section ***\n\n")
# Print out the encoded target list
print("\n\t *** Encoded target list ***\n")
print(encoded_target_array)
print("_" * 50)

input("Press Enter to Continue...")

# Split data into training and testing
features_training_data, features_testing_data, target_training_data, target_testing_data = train_test_split(encoded_feature_array, encoded_target_array, test_size=0.21, random_state=32)

# Prepare a decision tree classifier
classifier_dt = tree.DecisionTreeClassifier()

# Train the classifier
classifier_dt = classifier_dt.fit(features_training_data, target_training_data)

# Test the classifier
target_predicted_data = classifier_dt.predict(features_testing_data)

# Use the accuracy score feature from sklearn to see score
# Round the accuracy score for a better UX
rounded_score = round(accuracy_score(target_testing_data, target_predicted_data), 2)
# Turn into %
percent_score = rounded_score * 100

# Print out the accuracy score %
print("\n\t *** We will now show the accuracy score of the machine learning program based on the Mushroom Classification from Kaggle ***\n")
print("\n\n\tAccuracy Score: ", percent_score,"%")
print("\n\n")