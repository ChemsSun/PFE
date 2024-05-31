import numpy as np
import pandas as pd
import pickle
import streamlit as st
from collections import Counter

st.title('Données de Test')

# df=pd.read_csv('/content/x_test.csv')
# df1=pd.read_csv('/content/y_test.csv')
# # st.write(df)
# # st.write(df1)
# df_combined = pd.concat([df, df1], axis=1)
# st.write(df_combined)

df = pd.read_csv('/content/x_test.csv')
df1 = pd.read_csv('/content/y_test.csv')
df.columns = ['flag', 'src_bytes', 'dst_bytes', 'same_srv_rate', 'dst_host_same_srv_rate']
df1.columns = ['target']
df1.columns = [f"{col}_y" for col in df1.columns]

df_combined = pd.concat([df, df1], axis=1)
st.write(df_combined)
# Définir la classe OptunaKNN
class OptunaKNN:
    def __init__(self, k):
        self.k = k
        self.distances = []

    def fit(self, X_train, y_train):
        self.X_train = X_train  
        self.y_train = y_train

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        np.save('distances.npy', distances)
        k_indices = np.argsort(distances)[:self.k]
        np.save('k_indices.npy', k_indices)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        np.save('k_nearest_labels.npy', k_nearest_labels)
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

# Charger le modèle après avoir défini la classe
model_load = pickle.load(open('hoknn_model.pkl', 'rb'))

def intrusion_detection(input_data):
    # Convertir toutes les entrées en valeurs numériques (float)
    input_data = [float(i) for i in input_data]

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model_load.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'attaque'
    else:
        return 'Normal'

def main():
    st.title('IDS avec HO-KNN')
    Flag = st.text_input('Statut de connexion')
    src_bytes = st.text_input('Octets envoyés par la source')
    dst_bytes = st.text_input('Octets reçus par la destination')
    same_srv_rate = st.text_input('Taux de requêtes au même service')
    dst_host_same_srv_rate = st.text_input('Taux de requêtes au même service sur l hôte de destination')

    prd = ''
    if st.button('IDS test results'):
        prd = intrusion_detection([Flag, src_bytes, dst_bytes, same_srv_rate, dst_host_same_srv_rate])
    st.success(prd)

if __name__ == '__main__':
    main()
