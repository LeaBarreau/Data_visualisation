# Auteurs : Lauraly Montuelle, Léa Barreau
# M2 Actuariat 
# Projet Data Visualisation

## PRESENTATION
### CONTEXTE

L'objectif de ce projet est de réaliser différentes analyses graphiques sur une base de données préalablement selectionée. Ces analyses graphiques seront complètées par une analyse de dépendance entre une variable cible et des variables explicatives de la base, à l'aide de modèles de machine learning. 
		
La base utilisée comporte des données sur les accidents de vélo en France entre 2005 et 2021. En complément d'informations sur les la gravité des blessures de l'individus, nous retrouvons la localisation de l'accident (dans le temps et l'espace), ainsi que les conditions dans lesquels l'accident s'est produit (météo, surface, équipement,...). Nous chercherons donc à expliquer la gravité des blessures des individus à partir des informations sur leurs accidents. 


### ARBORESCENCE
#### ARBORESCENCE GLOBALE
```
PROJET_DEEP_LEARNING/
├── accidentsVelo.xslx : Base de données retraitée sous format excel des accidents de vélo
├── Notebook.ipynb : Notebook qui permet d'afficher les graphiques d'analyse de la base de données
├── test.py : Script python qui permet de lancer l'application streamlit
├── README.md: Informations générales sur le projet
├── requirements.txt: Les packages ayant été utilisés pour le projet
```

### PRE-REQUIS
Pour exécuter les programmes, vous aurez besoin des packages contenus dans le fichier requirements.txt. Vous pouvez tous les installer avec la commande `pip install -r requirements.txt`


### LANCEMENT DES SCRIPTS
Se positionner dans le répertoire et lancer une console avec l'environnement adapté

```
%run test.py
```
