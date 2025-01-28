from sklearn.model_selection import train_test_split
from data_processing import DataProcessor
from model_trainer import ModelTrainer
import numpy as np


def main():
    data = DataProcessor.read_data('data\\raw\\mini_gm_public_v0.1.p')
    data = DataProcessor.data_to_pandas_df(data)

    DataProcessor.save_csv_data(data)

    X = np.array(data['embedding'].tolist())
    y = data['syndrome_id']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Preprocessing step
    processor = DataProcessor(X_train, y_train)
    X_train_normalized = processor.normalize_data() #normalizing the data to use in the model with 'cosine' distance

    #Training the model
    distances = ['euclidean', 'cosine']

    for distance_metric, X_input in zip(distances, [X_train, X_train_normalized]):
        model_trainer = ModelTrainer(X_input, y_train)
        model_trainer.train_knn(distance_metric, X_input)
    
        #Predicting
        y_pred, y_proba = model_trainer.predict(X_test)
        
        #Evaluating the model
        model_trainer.evaluate_model(distance_metric, y_test, y_pred, y_proba, y_train)
        #results
        model_trainer.get_results()



if __name__ == "__main__": 
    main()