import os
from fuzzywuzzy import process
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


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


def ensure_objects_folder():
    os.makedirs("./objects", exist_ok=True)