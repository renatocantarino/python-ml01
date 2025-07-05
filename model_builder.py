import pandas as pd
from datetime import datetime
import numpy as np
import random as py_random
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import const
from service import fetch_data_from_db
from utils import *
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# cria rede neural
import tensorflow as tf


# reprodutividade
seed = 41
np.random.seed(seed)
py_random.seed(seed)
tf.random.set_seed(seed)

df = fetch_data_from_db(const.consulta_sql)

df["idade"] = df["idade"].astype(int)
df["valorsolicitado"] = df["valorsolicitado"].astype(float)
df["valortotalbem"] = df["valortotalbem"].astype(float)


substitui_nulos(df)
corrigir_erros_digitacao(df, "profissao", const.profissoes_validas)

# Trata Outliers
df = tratar_outliers(df, "tempoprofissao", 0, 70)
df = tratar_outliers(df, "idade", 0, 110)


# Feature Engineering
df["proporcaosolicitadototal"] = (df["valorsolicitado"] / df["valortotalbem"]).astype(
    float
)


# splt train test
X = df.drop("classe", axis=1)
y = df["classe"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# normaliza os dados
X_test = save_scalers(
    X_test,
    [
        "tempoprofissao",
        "renda",
        "idade",
        "dependentes",
        "valorsolicitado",
        "valortotalbem",
        "proporcaosolicitadototal",
    ],
)
X_train = save_scalers(
    X_train,
    [
        "tempoprofissao",
        "renda",
        "idade",
        "dependentes",
        "valorsolicitado",
        "valortotalbem",
        "proporcaosolicitadototal",
    ],
)

mapeamento = {"ruim": 0, "bom": 1}
y_train = np.array([mapeamento[item] for item in y_train])
y_test = np.array([mapeamento[item] for item in y_test])
X_train = save_encoders(
    X_train,
    ["profissao", "tiporesidencia", "escolaridade", "score", "estadocivil", "produto"],
)
X_test = save_encoders(
    X_test,
    ["profissao", "tiporesidencia", "escolaridade", "score", "estadocivil", "produto"],
)


# Seleção de Atributos
model = RandomForestClassifier()

# Instancia o RFE
selector = RFE(model, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

# Transforma os dados
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
joblib.dump(selector, "./objects/selector.joblib")


keras_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# otimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
keras_model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# treinar
keras_model.fit(
    X_train,
    y_train,
    validation_split=0.2,  # 20% dos dados de treino para validação
    epochs=500,
    batch_size=10,
    verbose=1,
)


# salvar modelo
keras_model.save("./objects/model.keras")

y_pred = keras_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# modelo avaliado

print("Classification Report:")
keras_model.evaluate(X_test, y_test)

#metricas
print("metrics Report:")
print(classification_report(y_test, y_pred))