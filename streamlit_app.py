import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm

st.write('# Prevendo o preço de casas no Brasil')
st.write('---')

st.header('Selecione os atributos desejados')


area = st.slider('Área', min_value = 0, max_value = 1000)
quartos = st.slider('Quartos', min_value = 0, max_value = 20)
banheiros = st.slider('Banheiros', min_value = 0, max_value = 10)
vagas = st.slider('Vagas', min_value = 0, max_value = 12)


features = {
    'area': area,
    'quartos': quartos,
    'banheiros': banheiros,
    'vagas': vagas
}

features_df = pd.DataFrame([features], index=[0])

data_house = pd.read_csv('dataset_house_prices_in_brazil.csv')

X = data_house.drop('valor_total', axis = 1)
y = data_house['valor_total']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.30,
                                                    random_state = 42)

st.write(features_df)

model_lr = lm.LinearRegression()

model_lr.fit(X_train, y_train)
prediction = np.round(model_lr.predict(features_df), decimals=2)
menor_valor = prediction - 2171
maior_valor = prediction + 2171

if st.button('Prever preço do imóvel'):
    st.write('O valor estimado do imóvel é de BRL{}'.format(prediction))
    st.write('O modelo está errando R$2171,00 à mais ou à menos. Então a faixa de preço varia de BRL{} à BRL{}'. format(menor_valor, maior_valor))



