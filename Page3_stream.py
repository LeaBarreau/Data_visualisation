import pandas 
import folium
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import plotly.graph_objects as go
import geopandas as gpd
from streamlit_folium import folium_static
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import shap
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

# Définissez la largeur de la page Streamlit
st.set_page_config(layout="wide")

st.title("La sécurité routière, même à vélo !")
st.write("Les données expoitées courent depuis 2005")

# Utilisez pd.read_excel() pour lire le fichier Excel dans un DataFrame
@st.cache_data
def import_data():
    data = pandas.read_csv(r"Base_VO.csv", decimal=".", sep=",")  
    data = data.loc[data['an'] >= 2015]
    # Créer une correspondance département - région
    departement_region = {
        1: 'Auvergne-Rhône-Alpes',
        2: 'Hauts-de-France',
        3: 'Auvergne-Rhône-Alpes',
        4: 'Provence-Alpes-Côte d\'Azur',
        5: 'Provence-Alpes-Côte d\'Azur',
        6: 'Provence-Alpes-Côte d\'Azur',
        7: 'Auvergne-Rhône-Alpes',
        8: 'Grand Est',
        9: 'Occitanie',
        10: 'Grand Est',
        11: 'Occitanie',
        12: 'Occitanie',
        13: 'Provence-Alpes-Côte d\'Azur',
        14: 'Normandie',
        15: 'Auvergne-Rhône-Alpes',
        16: 'Nouvelle-Aquitaine',
        17: 'Nouvelle-Aquitaine',
        18: 'Centre-Val de Loire',
        19: 'Nouvelle-Aquitaine',
        21: 'Bourgogne-Franche-Comté',
        22: 'Bretagne',
        23: 'Nouvelle-Aquitaine',
        24: 'Nouvelle-Aquitaine',
        25: 'Bourgogne-Franche-Comté',
        26: 'Auvergne-Rhône-Alpes',
        27: 'Normandie',
        28: 'Centre-Val de Loire',
        29: 'Bretagne',
        '2A': 'Corse',
        '2B': 'Corse',
        30: 'Occitanie',
        31: 'Occitanie',
        32: 'Occitanie',
        33: 'Nouvelle-Aquitaine',
        34: 'Occitanie',
        35: 'Bretagne',
        36: 'Centre-Val de Loire',
        37: 'Centre-Val de Loire',
        38: 'Auvergne-Rhône-Alpes',
        39: 'Bourgogne-Franche-Comté',
        40: 'Nouvelle-Aquitaine',
        41: 'Centre-Val de Loire',
        42: 'Auvergne-Rhône-Alpes',
        43: 'Auvergne-Rhône-Alpes',
        44: 'Pays de la Loire',
        45: 'Centre-Val de Loire',
        46: 'Occitanie',
        47: 'Nouvelle-Aquitaine',
        48: 'Occitanie',
        49: 'Pays de la Loire',
        50: 'Normandie',
        51: 'Grand Est',
        52: 'Grand Est',
        53: 'Pays de la Loire',
        54: 'Grand Est',
        55: 'Grand Est',
        56: 'Bretagne',
        57: 'Grand Est',
        58: 'Bourgogne-Franche-Comté',
        59: 'Hauts-de-France',
        60: 'Hauts-de-France',
        61: 'Normandie',
        62: 'Hauts-de-France',
        63: 'Auvergne-Rhône-Alpes',
        64: 'Nouvelle-Aquitaine',
        65: 'Occitanie',
        66: 'Occitanie',
        67: 'Grand Est',
        68: 'Grand Est',
        69: 'Auvergne-Rhône-Alpes',
        70: 'Bourgogne-Franche-Comté',
        71: 'Bourgogne-Franche-Comté',
        72: 'Pays de la Loire',
        73: 'Auvergne-Rhône-Alpes',
        74: 'Auvergne-Rhône-Alpes',
        75: 'Île-de-France',
        76: 'Normandie',
        77: 'Île-de-France',
        78: 'Île-de-France',
        79: 'Nouvelle-Aquitaine',
        80: 'Hauts-de-France',
        81: 'Occitanie',
        82: 'Occitanie',
        83: 'Provence-Alpes-Côte d\'Azur',
        84: 'Provence-Alpes-Côte d\'Azur',
        85: 'Pays de la Loire',
        86: 'Nouvelle-Aquitaine',
        87: 'Nouvelle-Aquitaine',
        88: 'Grand Est',
        89: 'Bourgogne-Franche-Comté',
        90: 'Bourgogne-Franche-Comté',
        91: 'Île-de-France',
        92: 'Île-de-France',
        93: 'Île-de-France',
        94: 'Île-de-France',
        95: 'Île-de-France',
        971: 'Guadeloupe',
        972: 'Martinique',
        973: 'Guyane',
        974: 'La Réunion',
        976: 'Mayotte'
    }
    data['region'] = data['dep'].map(departement_region)
    data = data.dropna(subset=['region'])

    region_mapping = {
        'Auvergne-Rhône-Alpes': 1,
        'Hauts-de-France': 2,
        'Provence-Alpes-Côte d\'Azur': 3,
        'Grand Est': 4,
        'Occitanie': 5,
        'Normandie': 6,
        'Centre-Val de Loire': 7,
        'Bourgogne-Franche-Comté': 8,
        'Bretagne': 9,
        'Corse': 10,
        'Pays de la Loire': 11,
        'Nouvelle-Aquitaine': 12,
        'Île-de-France': 13,
        'Guadeloupe': 14,
        'Martinique': 15,
        'Guyane': 16,
        'La Réunion': 17,
        'Mayotte': 18
    }

    # Appliquer la correspondance à la colonne 'region' de votre DataFrame
    data['region2'] = data['region'].map(region_mapping)
    
    data['grav'] = 0*data['indemne'] + 2*data['blesse_hospi'] + 1*data['hospi_leger'] + 3*data['tue']
    data['equipement'] = data['equipement'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    return data

data = import_data()

@st.cache_data
def test_model():
    # Séparer les features et la variable cible
    X = data[["secuexist", "age", "region2", "lum", "atm", "catr", "trajet", "equipement"]]
    y = data['grav']

    # Imputer les valeurs manquantes
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Diviser les données en ensembles d'entraînement et de test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(),
        'SVM': SVC()
    }
    # Créer un DataFrame pour stocker les importances de chaque modèle
    importances_df = pd.DataFrame(index=X_imputed.columns)
    # Boucle pour entraîner et évaluer chaque modèle
    results = {}
    class_accuracies = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        # Calculer des métriques de performance 
        accuracy = accuracy_score(y_test, y_pred)
        # Extraire la matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Calculer le pourcentage de prédictions correctes par classe
        class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        # Stocker les résultats
        results[model_name] = {'model': model, 'accuracy': accuracy}
        class_accuracies[model_name] = {'class_accuracy': class_accuracy}

    # Créer une figure avec 4 sous-plots disposés en ligne
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Afficher la matrice de confusion pour chaque modèle
    for ax, (model_name, model_conf_matrix) in zip(axes, zip(class_accuracies.keys(), [confusion_matrix(y_test, model.predict(X_test)) for model in models.values()])):
        ax.set_title(f"Matrice de Confusion pour {model_name}")
        
        # Calculer les pourcentages
        total_per_class = model_conf_matrix.sum(axis=1)
        conf_matrix_percentage = (model_conf_matrix.T / total_per_class).T * 100

        # Afficher une heat map de la matrice de confusion
        sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Valeurs Prédites')
        ax.set_ylabel('Valeurs Réelles')

    # Ajuster l'espacement entre les sous-plots
    plt.tight_layout()

    # Afficher la figure
    st.pyplot(fig)

    # Créer un DataFrame pour les accuracies
    accuracies_df = pd.DataFrame(results).T  # Transposer pour avoir les modèles en index
    accuracies_df.reset_index(inplace=True)
    accuracies_df.columns = ['Model', 'Data']
    accuracies_df[['Model', 'Accuracy']] = pd.DataFrame(accuracies_df['Data'].tolist(), index=accuracies_df.index)
    accuracies_df = accuracies_df[['Model', 'Accuracy']]

    # Appliquer une mise en forme conditionnelle pour mettre en couleur la ligne du modèle XGBoost
    styled_df = accuracies_df.style.apply(lambda x: ['background: lightblue' if x['Model'] == 'XGBoost' else '' for i in x], axis=1)

    # Afficher le DataFrame stylé
    st.dataframe(styled_df, use_container_width=True)

    st.write("Nous choisissons le modèle XGboost pour notre application.")

    # Entraîner le modèle XGBoost
    xgb_model = models['XGBoost']
    xgb_model.fit(X_train, y_train.values.ravel())

    # Obtenir les importances des caractéristiques du modèle XGBoost
    importances_df['XGBoost'] = xgb_model.feature_importances_

    # Afficher les importances des variables pour le modèle choisi (XGBoost)
    chosen_model_importances = importances_df['XGBoost']
    # Afficher un graphique à barres pour les importances des variables du modèle choisi (XGBoost)
    fig1 = px.bar(x=chosen_model_importances.index, y=chosen_model_importances.values, title='Test')
    fig1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title='Caractéristiques'),
        yaxis=dict(showgrid=False, title='Importance')
    )
    st.plotly_chart(fig1, use_container_width=True, figsize=(10, 6))

test_model()

st.write("Nous remarquons pour chaque modèle que les classes 0 et 3 ne sont pas bien prédites en raison de leur faible représentation dans la base de données.")
st.write("Pour vaincre cela, nous allons rassemblé les classes 0 et 1 et les classe 2 et 3.")

@st.cache_data
def xgboost_merged_classes():
    # Séparer les features et la variable cible
    X = data[['age', 'region2', 'catr', 'trajet']]
    y = data['grav']

    # Fusionner les classes 0 et 1 en une seule classe (classe 0)
    # Fusionner les classes 2 et 3 en une seule classe (classe 1)
    y_merged = y.replace({0: 0, 1: 0, 2: 1, 3: 1})

    # Diviser les données en ensembles d'entraînement et de test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y_merged, test_size=0.2, random_state=42, stratify=y_merged)

    # Définir les hyperparamètres à optimiser
    param_dist = {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 1, 2],
        'min_child_weight': [1, 2, 3],
        'lambda': [0, 1, 2],
        'alpha': [0, 1, 2]
    }

    # Créer le modèle XGBoost
    xgb = XGBClassifier()

    # Effectuer une recherche aléatoire des hyperparamètres
    random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=42)
    random_search.fit(X_train, y_train)

    # Afficher les meilleurs hyperparamètres
    best_params = random_search.best_params_
    st.write("Meilleurs hyperparamètres :", best_params)

    # Utiliser le modèle avec les meilleurs hyperparamètres pour prédire sur les données de test
    y_pred = random_search.best_estimator_.predict(X_test)

    # Calculer et afficher l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy sur les données de test :")
    st.write(accuracy)

    # Afficher la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write("Matrice de Confusion :")
    st.write(conf_matrix)

xgboost_merged_classes()

# Liste des régions françaises métropolitaines
regions_francaises = [
    'Auvergne-Rhône-Alpes',
    'Bourgogne-Franche-Comté',
    'Bretagne',
    'Centre-Val de Loire',
    'Corse',
    'Grand Est',
    'Hauts-de-France',
    'Île-de-France',
    'Normandie',
    'Nouvelle-Aquitaine',
    'Occitanie',
    'Pays de la Loire',
    'Provence-Alpes-Côte d\'Azur'
]

# Section pour la sélection de la région
st.header('Sélectionnez une région:')
selected_region = st.selectbox('Choisissez une région:', regions_francaises)

# Section pour la sélection du sexe et de l'âge
st.header('Sélectionnez le sexe et l\'âge:')
gender_options = ['Masculin', 'Féminin']
selected_gender = st.radio('Choisissez le sexe:', gender_options)

age_options = ['Moins de 18 ans', '18-25 ans', '26-35 ans', '36-50 ans', 'Plus de 50 ans']
selected_age = st.selectbox('Choisissez une tranche d\'âge:', age_options)

# Affichage des résultats
st.write('Vous avez choisi la région:', selected_region)
st.write('Vous avez choisi le sexe:', selected_gender)
st.write('Vous avez choisi la tranche d\'âge:', selected_age)