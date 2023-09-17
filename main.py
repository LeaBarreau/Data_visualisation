import streamlit as st
import pandas as pd
import time
st.set_page_config(page_title="Reporting KYF", page_icon="🔍", layout="wide")
annee=st.sidebar.selectbox("Choix de l'année",options=["2022","2021"], help="choisissez l'année ")
type = st.sidebar.radio("Choix du type", ("Global","Scénario","Stock"))
st.write(type)
st.markdown(    """    <style>    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {        width: 220px;    }    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {        width: 220px;        margin-left: -220px;    }    </style>    """,    unsafe_allow_html=True,)
st.markdown(f'<p style="background-color:#F14938; color:white; margin: -8%; padding-left: 4%; font-size: 176%;"> Reporting KYF</p>', unsafe_allow_html=True)
data = {'product_name': ['laptop', 'printer', 'tablet', 'desk', 'chair'],        'price': [1200, 150, 300, 450, 200]        }
df = pd.DataFrame(data)
st.write(df)
st.title("Première App avec Streamlit")
st.header('Sous-titre')