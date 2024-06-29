import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import Flask, request, jsonify
import pickle
import os

class IrisFlowerClassifier:
    def __init__(self):
        self.iris_data  = None
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.model_filename = 'iris_flower_classification_model.pkl'
        self.scaler_filename = 'iris_flower_classification_scaler.pkl'

        self.base_dir = 'results/'
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.eda_dir = 'EDA_results/'
        if not os.path.exists(self.eda_dir):
            os.mkdir(self.eda_dir)


    def load_dataset(self):
      self.iris_data = datasets.load_iris()

    def eda(self):

        df = pd.DataFrame(data=np.c_[self.iris_data['data'], self.iris_data['target']], columns=self.iris_data['feature_names'] + ['target'])
        
        # Display the first few rows of the dataset
        print(df.head())

        # Summary statistics
        print(df.describe())

        # Pairplot
        sns.pairplot(df, hue='target', palette='colorblind')
        plt.savefig(os.path.join(self.eda_dir, 'pairplot.png'))
        plt.show()
        
        # Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(self.eda_dir, 'Correlation Heatmap'))
        plt.show()
        
        # Distribution of features
        df.hist(bins=20, figsize=(12, 10), color='steelblue', edgecolor='black')
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'feature_distributions.png'))
        plt.show()
        

    def split_dataset(self):
        X = self.iris_data.data
        y = self.iris_data.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self):

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
    def save_model(self):
        with open(self.model_filename, 'wb') as f:
            pickle.dump(self.model, f)

        with open(self.scaler_filename, 'wb') as g:
            pickle.dump(self.scaler, g)
    

    def evaluate_model(self):
        self.X_test = self.scaler.transform(self.X_test)
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        
        self.predictions = y_pred 

        return accuracy, report, matrix
    
    def save_result_metrices(self,accuracy, report, matrix):
        result_dict = {
        "Accuracy":accuracy,
        "report":report,
        "conf_matrix":matrix
       }

        results_path = os.path.join(self.base_dir, 'result.txt')
        with open(results_path,"w") as file1:
            for k,v in result_dict.items():
                file1.write("{} : {} \n".format(k,v))

    def print_results(self,accuracy, report, matrix):
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        print(f"Confusion Matrix:\n{matrix}")

    def plot_results(self,matrix):
        plt.figure(figsize=(10,7))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        plt.show()

    def run(self):
        # Execute all steps in sequence
        self.load_dataset()
        self.eda()
        self.split_dataset()
        self.train_model()
        accuracy, report, matrix = self.evaluate_model()
        self.save_result_metrices(accuracy, report, matrix)
        self.print_results(accuracy, report, matrix)
        self.save_model()
        self.plot_results(matrix)
    

if __name__ == "__main__":
    classifier = IrisFlowerClassifier()
    classifier.run()