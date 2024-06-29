import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

eda_dir = 'EDA_results/'

# Load the dataset
data_path =  "car data.csv"
df = pd.read_csv(data_path)

# Number of rows and columns in the dataset
print(df.shape)

# Basic statistics and info
print("Dataset information:")
print(df.info())

#Finding missing values
print(df.isnull().sum())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Distribution of numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(eda_dir, 'Distribution of features.png'))
plt.close()

# Boxplot of numerical features
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_features])
plt.title('Boxplot of Numerical Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()
plt.savefig(os.path.join(eda_dir, 'Boxplot of Numerical Features'))
plt.close()

# Pairplot for numerical features
sns.pairplot(df[numerical_features])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()
plt.savefig(os.path.join(eda_dir, 'Pairplot of Numerical Features'))
plt.close()

# Count plot for categorical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
plt.figure(figsize=(12, 6))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, 3, i)
    sns.countplot(data=df, x=feature)
    plt.title(f'Count of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(eda_dir, 'Total count of Features'))
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
plt.savefig(os.path.join(eda_dir, 'Correlation Heatmap Between features'))
plt.close()
