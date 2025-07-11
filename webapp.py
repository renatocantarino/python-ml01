import streamlit as st
import requests

import const_web
from utils import load_yaml_config

st.title("Risk Assessment .: ML Web App :. ")

config = load_yaml_config()
url = config["url_api"]["url"]


def validate_values(value, array_errors):
    if value <= 0:
        array_errors.append("Valor solicitado deve ser maior que zero.")

    return array_errors


with st.form(key="prediction_form"):
    profissao = st.selectbox("Profissão", const_web.profissoes)
    tempo_profissao = st.number_input(
        "Tempo na profissão (em anos)", min_value=0, value=0, step=1
    )
    renda = st.number_input("Renda mensal", min_value=0.0, value=0.0, step=1000.0)
    tipo_residencia = st.selectbox("Tipo de residência", const_web.tipos_residencia)
    escolaridade = st.selectbox("Escolaridade", const_web.escolaridades)
    score = st.selectbox("Score", const_web.scores)
    idade = st.number_input("Idade", min_value=18, max_value=110, value=25, step=1)
    dependentes = st.number_input("Dependentes", min_value=0, value=0, step=1)
    estado_civil = st.selectbox("Estado Civil", const_web.estados_civis)
    produto = st.selectbox("Produto", const_web.produtos)
    valor_solicitado = st.number_input(
        "Valor solicitado", min_value=0.0, value=0.0, step=1000.0
    )
    valor_total_bem = st.number_input(
        "Valor total do bem", min_value=0.0, value=0.0, step=1000.0
    )

    submit_button = st.form_submit_button(label="Consultar")

if submit_button:
    errors = []
    validate_values(valor_solicitado, errors)
    validate_values(valor_total_bem, errors)  

    if errors:
        st.error("Erros encontrados:")
        for error in errors:
            st.error(error)
    else:
        dados_novos = {
            "profissao": [profissao],
            "tempoprofissao": [tempo_profissao],
            "renda": [renda],
            "tiporesidencia": [tipo_residencia],
            "escolaridade": [escolaridade],
            "score": [score],
            "idade": [idade],
            "dependentes": [dependentes],
            "estadocivil": [estado_civil],
            "produto": [produto],
            "valorsolicitado": [valor_solicitado],
            "valortotalbem": [valor_total_bem],
            "proporcaosolicitadototal": [valor_total_bem / valor_solicitado],
        }

        response = requests.post(url, json=dados_novos)
        if response.status_code == 200:
            predictions = response.json()            
            probabilidade = predictions['data'][0][0] * 100
            classe = "Bom" if probabilidade > 50 else "Ruim"
            st.success(f"Probabilidade: {probabilidade:.2f}%")
            st.success(f"Classe: {classe}")
        else:
            st.error(f"Erro ao fazer a previsão: {response.status_code}")

