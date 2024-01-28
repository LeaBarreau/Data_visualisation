# Auteurs : Lauraly Montuelle, Léa Barreau
# M2 Actuariat 
# Projet Data Visualisation

## PRESENTATION
### CONTEXTE

L'objectif de ce projet est de réaliser différentes analyses graphiques sur une base de données préalablement selectionée. Ces analyses graphiques seront complètées par une analyse de dépendance entre une variable cible et des variables explicatives présentes dans la base, à l'aide de modèles de machine learning. 
		
La base utilisée ici comporte des données sur les accidents de vélo en France entre 2005 et 2021. En complément d'informations sur la gravité des blessures de l'individu, nous retrouvons la localisation de l'accident (dans le temps et l'espace), ainsi que les conditions dans lesquels l'accident s'est produit (météo, surface, équipement,...). Nous chercherons de plus à expliquer la gravité des blessures des individus à partir des informations sur leur accident, tout en donnant un apercu visuel de la sinistralité vélo en France.


### ARBORESCENCE
#### ARBORESCENCE GLOBALE
```
PROJET_DATA_VISUALISATION/
├── accidentsVelo.xslx : Base de données retraitée des accidents de vélo en France
├── Base_dep.csv : Base des sinistres à la maille département
├── Base_VO.csv : Base de données d'origine utilisée pour la partie machine learning
├── main.py : Script permettant de créer l'application streamlit
├── App_streamlit.py : Script permettant de lancer les différentes pages de l'application streamlit
├── Page1_stream.py : Script permettant d'obtenir la partie analyse descriptive de l'étude
├── Page2_stream.py : Script concernant la partie interactive de l'étude à la maille département / région
├── Page3_stream.py : Script centré sur la partie machine learning du projet
├── README.md: Informations générales sur le projet
├── requirements.txt: Les packages ayant été utilisés pour le projet
```

### PRE-REQUIS
Pour exécuter les programmes, vous aurez besoin des packages contenus dans le fichier requirements.txt. Vous pouvez tous les installer avec la commande `pip install -r requirements.txt`


### LANCEMENT DES SCRIPTS
Se positionner dans le répertoire et lancer une console avec l'environnement adapté

```
streamlit run App_streamlit.py
```
