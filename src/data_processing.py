import pickle 
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, X, y):
        self.X = X
        self. y = y
        self.scaler = StandardScaler()

    def read_data(path):
        """
        Reading the pickle file
        """
        with open(path, 'rb') as file:
            data = pickle.load(file)
            return data

    def data_to_pandas_df(data):
        """"
        Coverting the data to pandas dataframe
        """
        data_flat = []

        for syndrome_id, subjects in data.items():
            for subject_id, images in subjects.items():
                for image_id, embedding in images.items():
                    data_flat.append({
                        'syndrome_id': syndrome_id,
                        'subject_id': subject_id,
                        'image_id': image_id,
                        'embedding': embedding
                    })
        
        df = pd.DataFrame(data_flat)
        return df

    def save_csv_data(df):
        """
        Saving the dataframe as a csv file 
        """
        return df.to_csv('data/processed/processed_data.csv', index=False)
    
    def normalize_data(self):
        """
        Normalizing the data to use in the model with 'cosine' distance
        """
        X_normalized = self.scaler.fit_transform(self.X)
        return X_normalized
