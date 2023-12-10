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

# Séparer les features et la variable cible
X = data[["secuexist", "age", "region2", "lum", "atm", "catr", "trajet", "equipement"]]
#X = data[["age", "region2", "catr", "trajet"]]
y = data[["grav"]]

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Modèles à tester
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
for model_name, model in models.items():
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    
    # Calculer des métriques de performance 
    accuracy = accuracy_score(y_test, y_pred)
    
    # Extraire l'importance des caractéristiques
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances = model.coef_
    else:
        feature_importances = None
    
    # Stocker les résultats
    results[model_name] = {'model': model, 'accuracy': accuracy, 'feature_importances': feature_importances}
    
    # Stocker les importances dans le DataFrame
    importances_df[model_name] = feature_importances

# Créer un DataFrame à partir des résultats
results_df = pd.DataFrame(results).T
results_df = results_df[['accuracy']]  # Sélectionner les colonnes de métriques

# Afficher le tableau avec les métriques
st.write("Comparaison des modèles (Métriques de régression) :")
st.table(results_df)

# Afficher les importances des caractéristiques avec un graphique à barres
plt.style.use('dark_background')  # Utiliser un fond noir
importances_df.plot(kind='bar', figsize=(10, 6), title='Importance des Caractéristiques dans Chaque Modèle', xlabel='Caractéristiques', ylabel='Importance', colormap='viridis')

# Ajuster les paramètres du graphique pour améliorer la lisibilité
plt.xticks(rotation=45, ha='right')  # Rotation des étiquettes sur l'axe x
plt.tight_layout()  # Ajustement automatique de la mise en page pour éviter les coupures
plt.legend(loc='upper right')  # Ajouter une légende

# Afficher le graphique
st.pyplot(plt)


## CHOSIR XGBOOST ET OPTIMISER LES HYPERPARAMETRE POUR AVOIR UN MODELE TROP BIEN
# Séparer les features et la variable cible
X = data[['age', 'region2', 'catr', 'trajet']]
y = data['grav']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
print("Meilleurs hyperparamètres :", best_params)

# Utiliser le modèle avec les meilleurs hyperparamètres pour prédire sur les données de test
y_pred = random_search.best_estimator_.predict(X_test)

# Calculer et afficher l'accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy sur les données de test :")
st.write(accuracy)

# Séparer les features et la variable cible
X = data[["age", "region2", "catr", "trajet"]]
y = data['grav']

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Utiliser le modèle XGBoost avec les meilleurs hyperparamètres pour prédire sur l'ensemble initial de features
y_pred_best = random_search.best_estimator_.predict(X_test)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred_best, normalize='true')

# Afficher la matrice de confusion sous forme de heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Indemne', 'Blesse_hospi', "hospi_leger", "Tue"], yticklabels=['Indemne', 'Blesse_hospi', "hospi_leger", "Tue"])
plt.title('Matrice de Confusion - Modèle XGBoost')
plt.xlabel('Vrai')
plt.ylabel('Prédit')
plt.tight_layout()

# Afficher le graphique
st.pyplot(plt)

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