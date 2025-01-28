import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, top_k_accuracy_score, classification_report, RocCurveDisplay, accuracy_score
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, X, y):
        self. X = X
        self.y = y
        self.best_k = None
        self.best_score = 0
        self.results = None
    

    def train_knn(self, distance_metric, X_input, k_range=15):
        """
        Train the KNN model and get the best k value and the best score
        """
        
        self.results = {'metric': distance_metric,
                        'k': [],
                        'accuracy': [],
                        'auc': [],
                        'f1': [],
                        'top_5_accuracy': []}
        
        self.best_k = None
        self.best_score = 0

        for k in range(1, k_range + 1):
            knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
            scores = cross_val_score(knn, X_input, self.y, cv=10, scoring='accuracy')
            mean_score = scores.mean()

            self.results['k'].append(k)
            self.results['accuracy'].append(mean_score)

            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_k = k

        
        print(f" ======================= {distance_metric.upper()} DISTANCE =======================")
        print(f"\nMelhor valor de k: {self.best_k} com acurácia média de {self.best_score}\n")
    

    def predict(self, X_test):
        """
        Predict the labels for the test data
        """
        knn = KNeighborsClassifier(n_neighbors=self.best_k)
        knn.fit(self.X, self.y)

        y_pred = knn.predict(X_test)
        y_proba = knn.predict_proba(X_test)

        return y_pred, y_proba

    
    def evaluate_model(self, distance_metric, y_test, y_pred, y_proba, y_train):
        """
        Evaluate the model with the best k value
        """

        if self.best_k is None:
            raise ValueError("First of all, train the model using the train_knn() method.")

        #evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        top_5_acc = top_k_accuracy_score(y_test, y_proba, k=5)

        self.results['accuracy'].append(accuracy)
        self.results['f1'].append(f1)
        self.results['auc'].append(auc)
        self.results['top_5_accuracy'].append(top_5_acc)

        print(f"Results using {distance_metric} distance:")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"AUC: {auc}")
        print(f"Top-5 Accuracy: {top_5_acc}")   

        #ROC curve
        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test)    

        display = RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_proba.ravel(),
        name="micro-average OvR",
        color="darkorange",
        plot_chance_level=True,
        despine=True,
        )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Micro-averaged One-vs-Rest\nReceiver Operating Characteristic\n{distance_metric} distance",
        )

        plt.savefig(f"roc_curve_{distance_metric}.png")  #Saving the plot as a png file

    def get_results(self):
        """
        Return the evaluation results
        """

        if self.results is None:
            raise ValueError("No reuslts found. First of all, evaluate the model using the evaluate_model() method.")
        
        return self.results



            



