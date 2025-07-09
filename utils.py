import os
from fuzzywuzzy import process
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

import const


def substitui_nulos(df):
    for coluna in df.columns:
        if df[coluna].dtype == "object":
            moda = df[coluna].mode()[0]
            df[coluna].fillna(moda, inplace=True)
        else:
            mediana = df[coluna].median()
            df[coluna].fillna(mediana, inplace=True)


def corrigir_erros_digitacao(df, coluna, lista_valida):
    for i, valor in enumerate(df[coluna]):
        valor_str = str(valor) if pd.notnull(valor) else valor

        if valor_str not in lista_valida and pd.notnull(valor_str):
            correcao = process.extractOne(valor_str, lista_valida)[0]
            df.at[i, coluna] = correcao


def tratar_outliers(df, coluna, minimo, maximo):
    mediana = df[(df[coluna] >= minimo) & (df[coluna] <= maximo)][coluna].median()
    df[coluna] = df[coluna].apply(lambda x: mediana if x < minimo or x > maximo else x)
    return df


def save_scalers(df, nome_colunas):
    ensure_objects_folder()
    for nome_coluna in nome_colunas:
        scaler = StandardScaler()
        df[nome_coluna] = scaler.fit_transform(df[[nome_coluna]])
        joblib.dump(scaler, f"./objects/scaler{nome_coluna}.joblib")

    return df


def save_encoders(df, nome_colunas):
    ensure_objects_folder()
    for nome_coluna in nome_colunas:
        label_encoder = LabelEncoder()
        df[nome_coluna] = label_encoder.fit_transform(df[nome_coluna])
        joblib.dump(label_encoder, f"./objects/labelencoder{nome_coluna}.joblib")

    return df


def load_scalers(df, collum_names):
    for nome_coluna in collum_names:
        nome_arquivo_scaler = f"./objects/scaler{nome_coluna}.joblib"
        scaler = joblib.load(nome_arquivo_scaler)
        df[nome_coluna] = scaler.transform(df[[nome_coluna]])
    return df


def load_encoders(df, collum_names):
    for nome_coluna in collum_names:
        nome_arquivo_encoder = f"./objects/labelencoder{nome_coluna}.joblib"
        label_encoder = joblib.load(nome_arquivo_encoder)
        df[nome_coluna] = label_encoder.transform(df[nome_coluna])
    return df


def prepare_model_to_explain(
    data_asarray, keras_model, X_train_columns, scaler_features, encoder_features
):
    data_asframe = pd.DataFrame(data_asarray, columns=X_train_columns)
    data_asframe = save_scalers(data_asframe, scaler_features)
    data_asframe = save_encoders(data_asframe, encoder_features)
    predictions = keras_model.predict(data_asframe)
    return np.hstack((1 - predictions, predictions))


def ensure_objects_folder():
    os.makedirs("./objects", exist_ok=True)
