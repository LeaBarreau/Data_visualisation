import pandas 
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import sklearn
import pandas as pd

# Définissez la largeur de la page Streamlit
st.set_page_config(layout="wide")

st.title("🚗 La Sécurité Routière à Vélo : Explorer et Prévenir ! 🚴‍♂️")
st.write("Bienvenue sur cette page dédiée à la sécurité routière, mettant l'accent sur les accidents impliquant des cyclistes. Nous plongeons dans les données routières depuis 2005 pour comprendre, analyser et améliorer la sécurité des cyclistes sur nos routes. Nous allons entraîner un modèle de machine learning afin d'étudier l'impact de différents facteur sur la gravité des accients de vélo.")
st.write("📊 Les Données en Bref : Notre exploration couvre une période étendue, nous permettant de saisir les évolutions au fil des années. Ces données, issues de différentes sources, sont soigneusement analysées pour offrir des perspectives sur les accidents de la route impliquant des cyclistes.")
st.write("🌐 Contexte Régional : À travers notre analyse, nous explorerons les spécificités régionales, car chaque région a ses propres caractéristiques et défis en matière de sécurité routière.")
st.write("🚨 Objectif de cette page : Notre mission est de mieux comprendre les facteurs qui contribuent aux accidents de vélo, de prévoir les situations à risque et d'optimiser les modèles de prédiction pour renforcer la sécurité des cyclistes.")
st.write("Explorez avec nous les modèles, les tendances et les résultats qui émergent de cette analyse approfondie. Ensemble, travaillons à rendre nos routes plus sûres pour tous !")
st.write("Note : Nous concentrons notre analyse sur les années à partir de 2015, offrant ainsi une vision actuelle des enjeux de sécurité routière à vélo.")

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
    data = data.dropna(subset=['dep'])

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
def model(X_train, y_train, X_test, y_test):
    @st.cache_data
    def test_model(X_train, X_test, y_train, y_test):
        models = {
            'Random Forest': RandomForestClassifier(),
            'KNN': KNeighborsClassifier(),
            'XGBoost': XGBClassifier(),
            'SVM': SVC()
        }
        accuracies = {}
        matrices = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train.values.ravel())
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[model_name] = accuracy
            conf_matrix = confusion_matrix(y_test, y_pred)
            matrices[model_name] = conf_matrix
        
        return accuracies, matrices

    @st.cache_data
    def calculate_variable_importance(X_train, y_train):
        xgb_model = XGBClassifier()
        xgb_model.fit(X_train, y_train.values.ravel())
        importances_df = pd.DataFrame(index=X_train.columns)
        importances_df['XGBoost'] = xgb_model.feature_importances_
        return importances_df

    # Calculate variable importance
    importances_df = calculate_variable_importance(X_train, y_train)
    # Calculate accuracies and confusion matrices
    accuracies, confusion_matrices = test_model(X_train, X_test, y_train, y_test)

    # Display confusion matrices
    st.write("Affichons les matrices de confusion des différents modèles.")
    st.write("Une matrice de confusion est un moyen de visualiser où notre modèle a bien fonctionné et où il a eu des difficultés. Cela nous aide à comprendre comment un modèle se comporte dans différentes situations.")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (model_name, model_conf_matrix) in zip(axes, confusion_matrices.items()):
        ax.set_title(f"Matrice de Confusion pour {model_name}")
        total_per_class = model_conf_matrix.sum(axis=1)
        conf_matrix_percentage = (model_conf_matrix.T / total_per_class).T * 100
        sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Valeurs Prédites')
        ax.set_ylabel('Valeurs Réelles')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("Lors de l\'observation des matrices de confusion pour chaque modèle, il est évident que les classes 0 (Indemnes) et 3 (Tués) posent un défi de prédiction significatif. Les pourcentages associés à ces classes sont notablement bas, suggérant que les modèles ont du mal à les identifier correctement. Cette difficulté peut être attribuée à une représentation limitée de ces classes dans l\'ensemble de données, ce qui rend la généralisation plus complexe.")
    st.write("En revanche, les classes 1 (Blessés Légers) et 2 (Blessés hospitalisés) bénéficient d'une prédiction plus précise, avec des pourcentages plus élevés dans les matrices de confusion. Il semble que la représentation de ces classes soit plus robuste, facilitant ainsi la tâche des modèles.")

    # Display model accuracies
    st.write("En vue de prendre une décision quant au choix du modèle, il est nécessaire d'examiner les précisions de chacun d'entre eux.")
    df_accuracies = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy']).reset_index()
    st.dataframe(df_accuracies)
    st.write("Nous optons pour le modèle XGBoost dans notre application en raison de sa précision supérieure parmi les modèles testés. Afin de simplifier l'interprétation et d'accroître la stabilité du modèle, nous allons regrouper certaines classes de gravité. Les classes 'Indemne' et 'Blessé Léger' ont été fusionnées, de même que les classes 'Blessé Hospitalisé' et 'Tué'. Nous explorerons également l'importance de chaque variable dans le modèle sélectionné.")

    # Display variable importance
    st.write("Affichons l'importance des variables pour le modèle XGBoost.")
    fig1 = px.bar(x=importances_df.index, y=importances_df['XGBoost'], title='Importance des variables explicatives utilisées pour le modèle XGBoost')
    fig1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title='Caractéristiques'),
        yaxis=dict(showgrid=False, title='Importance')
    )
    st.plotly_chart(fig1, use_container_width=True, figsize=(10, 6))
    st.write("Nous notons que les variables 'Dep' (départements français) et 'Catr' (type de route) se démarquent comme les plus cruciales dans notre modèle XGBoost. Nous prévoyons de les maintenir pour la suite de l'analyse.")
    st.write("Nous avons choisi d'utiliser un modèle XGBoost, en raison de sa capacité à traiter des ensembles de données complexes et à fournir des prédictions précises. Après l'entraînement initial, nous allons optimisé les paramètres du modèle pour améliorer sa précision. Pour ce faire, nous allons tester plusieurs combinaisons d'hyperparamètres et conserver celle qui maximise la précision de notre modèle (accuracy)")


# Fonction pour fusionner les classes de gravité
def merge_gravity_classes(y):
    return y.replace({0: 0, 1: 0, 2: 1, 3: 1})
# Fonction pour diviser les données et fusionner les classes
def split_and_merge_data(data):
    X = data[['dep', 'catr', 'homme']]
    y = data['grav']
    y_merged = merge_gravity_classes(y)
    return train_test_split(X, y_merged, test_size=0.2, random_state=42, stratify=y_merged)
def split_and_merge_data2(data):
    X = data[["secuexist", "age", "dep", "lum", "atm", "catr", "trajet", "equipement", "homme"]]
    y = data['grav']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fonction pour effectuer une recherche aléatoire des hyperparamètres et retourner le modèle
def perform_random_search(X_train, y_train):
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
    xgb = XGBClassifier()
    random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    return best_params, best_model

# Fonction pour afficher la matrice de confusion sous forme de heatmap
def display_confusion_matrix_heatmap(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    total_per_class = conf_matrix.sum(axis=1)
    conf_matrix_percentage = (conf_matrix.T / total_per_class).T * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Valeurs Prédites')
    ax.set_ylabel('Valeurs Réelles')
    plt.title("Matrice de Confusion")
    st.pyplot(fig)

# Application des fonctions
    
X_train, X_test, y_train, y_test = split_and_merge_data(data)
X_train2, X_test2, y_train2, y_test2 = split_and_merge_data2(data)

X_train2_no_missing = X_train2.dropna()
y_train2_no_missing = y_train2.loc[X_train2_no_missing.index]
X_test2_no_missing = X_test2.dropna()
y_test2_no_missing = y_test2.loc[X_test2_no_missing.index]

best_params, best_model = perform_random_search(X_train, y_train)
y_pred = best_model.predict(X_test)
# Calcul et affichage de l'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Liste des régions françaises métropolitaines
departements = {
    1: "Ain",
    2: "Aisne",
    3: "Allier",
    4: "Alpes-de-Haute-Provence",
    5: "Hautes-Alpes",
    6: "Alpes-Maritimes",
    7: "Ardèche",
    8: "Ardennes",
    9: "Ariège",
    10: "Aube",
    11: "Aude",
    12: "Aveyron",
    13: "Bouches-du-Rhône",
    14: "Calvados",
    15: "Cantal",
    16: "Charente",
    17: "Charente-Maritime",
    18: "Cher",
    19: "Corrèze",
    21: "Côte-d'Or",
    22: "Côtes-d'Armor",
    23: "Creuse",
    24: "Dordogne",
    25: "Doubs",
    26: "Drôme",
    27: "Eure",
    28: "Eure-et-Loir",
    29: "Finistère",
    '2A': "Corse-du-Sud",
    '2B': "Haute-Corse",
    30: "Gard",
    31: "Haute-Garonne",
    32: "Gers",
    33: "Gironde",
    34: "Hérault",
    35: "Ille-et-Vilaine",
    36: "Indre",
    37: "Indre-et-Loire",
    38: "Isère",
    39: "Jura",
    40: "Landes",
    41: "Loir-et-Cher",
    42: "Loire",
    43: "Haute-Loire",
    44: "Loire-Atlantique",
    45: "Loiret",
    46: "Lot",
    47: "Lot-et-Garonne",
    48: "Lozère",
    49: "Maine-et-Loire",
    50: "Manche",
    51: "Marne",
    52: "Haute-Marne",
    53: "Mayenne",
    54: "Meurthe-et-Moselle",
    55: "Meuse",
    56: "Morbihan",
    57: "Moselle",
    58: "Nièvre",
    59: "Nord",
    60: "Oise",
    61: "Orne",
    62: "Pas-de-Calais",
    63: "Puy-de-Dôme",
    64: "Pyrénées-Atlantiques",
    65: "Hautes-Pyrénées",
    66: "Pyrénées-Orientales",
    67: "Bas-Rhin",
    68: "Haut-Rhin",
    69: "Rhône",
    70: "Haute-Saône",
    71: "Saône-et-Loire",
    72: "Sarthe",
    73: "Savoie",
    74: "Haute-Savoie",
    75: "Paris",
    76: "Seine-Maritime",
    77: "Seine-et-Marne",
    78: "Yvelines",
    79: "Deux-Sèvres",
    80: "Somme",
    81: "Tarn",
    82: "Tarn-et-Garonne",
    83: "Var",
    84: "Vaucluse",
    85: "Vendée",
    86: "Vienne",
    87: "Haute-Vienne",
    88: "Vosges",
    89: "Yonne",
    90: "Territoire de Belfort",
    91: "Essonne",
    92: "Hauts-de-Seine",
    93: "Seine-Saint-Denis",
    94: "Val-de-Marne",
    95: "Val-d'Oise",
    971: "Guadeloupe",
    972: "Martinique",
    973: "Guyane",
    974: "La Réunion",
    976: "Mayotte"
}
# Créer une liste des noms de départements
departement_names = list(departements.values())
# Créer une liste déroulante dans Streamlit avec les noms des départements
selected_departement_name = st.selectbox("Sélectionnez un département", departement_names)
# Convertir le nom de département en numéro en utilisant la liste departements
def get_departement_num(selected_departement_name):
    return [key for key, value in departements.items() if value == selected_departement_name][0]
# Section pour la sélection du sexe et de l'âge
st.header('Sélectionnez votre genre:')
gender_options = {'1': 'Masculin', '0': 'Féminin'}
selected_gender_name = st.radio('Choisissez le sexe:', list(gender_options.values()))
# Convertir le genre en numéro en utilisant le dictionnaire gender_options
selected_gender_num = [key for key, value in gender_options.items() if value == selected_gender_name][0]
# Liste des types de routes
st.header('Sélectionnez le type de routes que vous fréquentez le plus:')
catr_choice = {
    "1": 'Autoroute',
    "2": 'Route nationale',
    "3": 'Route départementale',
    "4": 'Voie communale',
    "5": 'Hors réseau public',
    "6": 'Parc de stationnement ouvert à la circulation publique',
    "7": 'Route de métropole urbaine',
    "9": 'Autre'
}
selected_catr_name = st.selectbox('Choisissez le type de route:', list(catr_choice.values()))
# Convertir le type de route en numéro en utilisant le dictionnaire catr_choice
selected_catr_num = [key for key, value in catr_choice.items() if value == selected_catr_name][0]
# Fonction pour obtenir les données d'entrée en fonction des sélections de l'utilisateur
def get_user_input(selected_departement_name, selected_gender_name, selected_catr_name):
    return pd.DataFrame({
        "dep": [get_departement_num(selected_departement_name)],
        "catr": [int(selected_catr_num)],
        "homme": [int(selected_gender_num)]
    })

# Fonction pour prédire et afficher les résultats
def predict_and_display_results(user_input):
    predicted_class = best_model.predict(user_input)[0]
    st.write("En fonction des sélections effectuées, le modèle de machine learning prédit la classe :", predicted_class)
    # Affichez la classe prédite
    st.write("En fonction des sélections effectuées, le modèle de machine learning prédit la classe :", predicted_class)
    if predicted_class == 0 :
        st.write("En d'autres termes, selon vos caractéristiques, notre modèle de machine learning prédit la classe 0. Cela signifie qu'en cas d'accident, il prévoit que vous resterez indemne ou vous serez blessé légèrement. Restez prudent ! ")
    else :
        st.write("En d'autres termes, selon vos caractéristiques, notre modèle de machine learning prédit la classe 1. Cela signifie qu'en cas d'accident, il prévoit que vous serez blessé gravement ou vous perdrez la vie. Restez prudent ! ")

    st.write("En plus de cela, nous avons quelques autres statistiques selon les caractéristiques inscrites :")

@st.cache_data
def import_data2():
    data2 = pd.read_csv(r"Base_dep.csv", decimal=".", sep=",")  
    return data2
data2 = import_data2()
bouton_test = st.button("Résultat avec mes données personnelles")

if bouton_test:
    st.write('Vous avez choisi la région:', selected_departement_name)
    st.write('Vous avez choisi le sexe:', selected_gender_name)
    st.write('Vous avez choisi le type de route:', selected_catr_name)

    # Préparez les données d'entrée en fonction des sélections de l'utilisateur
    user_input = get_user_input(selected_departement_name, selected_gender_name, selected_catr_name)
    # Faites une prédiction avec le modèle XGBoost et affichez les résultats
    predict_and_display_results(user_input)
    # Filtrer les données en fonction des sélections de l'utilisateur
    filtered_data = data2[(data2['dep'] == int(get_departement_num(selected_departement_name))) & (data2['catr'] == int(selected_catr_num)) & (data2['homme'] == int(selected_gender_num))]

    if not filtered_data.empty:
        st.write('Dans le cas d\'un accident, vous avez', filtered_data['part_indemne'].iloc[0]*100, '% de chance de rester indemne.')
        st.write('Dans le cas d\'un accident, vous avez', filtered_data['part_blesse_leger'].iloc[0]*100, '% de chance d\'être blessé légèrement.')
        st.write('Dans le cas d\'un accident, vous avez', filtered_data['part_blesse_hospi'].iloc[0]*100, '% de chance d\'être blessé et hospitalisé.')
        st.write('Dans le cas d\'un accident, vous avez', filtered_data['part_tue'].iloc[0]*100, '% de chance de perdre la vie.')
    else:
        st.write("Aucune donnée supplémentaire, nous sommes désolées :)")

# Ajoutez un espace dans la barre latérale
st.sidebar.write("Et pour les curieux :")

# Bouton dans la barre latérale
button_clicked = st.sidebar.button("Découvrez et comprenez notre modèle")

# Si le bouton est cliqué, affichez le contenu de la page spécifique
if button_clicked:
    st.title("Comprenons ensemble le modèle utilisé et l'optimisation de celui-ci !")
    st.header("Objectif du Modèle : ")
    st.write("Le modèle que nous avons développé a pour objectif de prédire la gravité des accidents de vélo en se basant sur divers facteurs. La gravité des accidents est classée en quatre catégories : indemne, blessé léger, blessé hospitalisé, et tué. Comprendre la gravité des accidents passés peut nous aider à identifier les principaux contributeurs aux incidents graves, ce qui à son tour peut informer des mesures de sécurité ciblées pour réduire les risques futurs sur nos routes.")
    st.header("Variables Utilisées : ")
    st.write(" - Securité Existante (secuexist) : Représente les équipements de sécurité portés par les individus impliqués dans l'accident (casque, ceinture, etc.).")
    st.write("- Âge (age) : L'âge des personnes impliquées dans l'accident.")
    st.write("- Région (region) : La région géographique où l'accident s'est produit. Chaque région a ses propres caractéristiques et défis en matière de sécurité routière.")
    st.write("- Luminosité (lum) : Les conditions d'éclairage au moment de l'accident.")
    st.write("- Atmosphère (atm) : Les conditions atmosphériques au moment de l'accident.")
    st.write("- Type de Route (catr) : La catégorie de route où l'accident s'est produit.")
    st.write("- Type de Trajet (trajet) : Le type de trajet effectué par les individus (domicile-travail, domicile-école, etc.).")
    st.write("- Équipement (equipement) : Les équipements spécifiques utilisés lors de l'accident.")
    st.write("- Homme (sexe) : Genre de l'individu accidenté.")
    st.header("Observons les résultats des différents modèles testés (KNN, SVM, Random Forest et XGBoost)")
    # Display the content of models presentation
    model(X_train2_no_missing, y_train2_no_missing, X_test2_no_missing, y_test2_no_missing)
    st.write("Voici la meilleurs combinaison d'hyperparamètres :", best_params)
    st.write("Avec cette combinaison d'hyperparamètres, nous obtenons l'accuracy suivante : ")
    st.write(accuracy)
    st.write("Nous constatons que la précision de notre modèle XGBoost, une fois optimisée, a augmenté de 10 points par rapport à la version non optimisée. Ainsi, nous atteignons désormais une précision d'environ 73%, ce qui se traduit par la capacité du modèle à correctement classer 73 individus sur 100.")
    st.write("Afin de confirmer notre choix, intéressons nous à la matrice de confusion :")
    # Affichage de la matrice de confusion sous forme de heatmap
    display_confusion_matrix_heatmap(y_test, y_pred)
    st.write("Le choix de notre modèle s'est avéré judicieux, comme le démontre la matrice de confusion. Cette dernière offre une visualisation claire de la performance du modèle en mettant en évidence un grand nombre de prédictions correctes et un nombre limité d'erreurs de prédiction. Les résultats indiquent une capacité significative du modèle à bien classifier les différentes classes de gravité des accidents de vélo. Nous observons une prépondérance de prédictions précises, illustrant ainsi la robustesse et la fiabilité de notre approche.")
        
