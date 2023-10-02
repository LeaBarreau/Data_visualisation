# Auteurs :    Barreau Léa, Paulin Anaïs, Pautonnier Chloé, Pichon Maxime
# M2 Actuariat Data Sciences
# Projet Deep Learning

## PRESENTATION
### CONTEXTE
Le but de ce projet est de construire des modèles de prédiction des caractéristiques des individus, notamment leur âge, leur sexe et leur ethnicité, pour les utiliser à des fins commerciales.

### ARBORESCENCE
#### ARBORESCENCE GLOBALE
```
PROJET_DEEP_LEARNING/
├── models/: Folder contenant les modèles
├── UTKFace/: Folder contenant les images du dataset UTKFace
├── cropped_UTKFace/: Folder contenant les images retravaillées du dataset UTKFace
├── cropped_pictures/: Folder contenant les résultats du script get_age_gender_ethnicity.py
├── .gitignore: Fichier pour déclarer ce qu'on suit sous git
├── Train_ResNet18_pretrained.ipynb : Notebook d'entrainement du ResNet18 préentrainé
├── CNN_age.ipynb : Notebook d'entrainement du modèle prédisant l'âge
├── CNN_gender.ipynb : Notebook d'entrainement du modèle prédisant le genre
├── CNN_ethnicity.ipynb : Notebook d'entrainement du modèle prédisant l'ethnicité
├── cropp_UTKFACE.ipynb : Notebook qui permet de traiter le UTKFace Dataset afin d'en extraire les visages
├── get_age_gender_ethnicity.py : Script python qui permet, pour une image placée dans le même folder, d'extraire les visages et donner l'age, le genre et l'ethnicité de chaque personne présente sur l'image
├── README.md: Votre serviteur
├── requirements.txt: Les packages ayant été utilisés pour le projet
```

### PRE-REQUIS
Pour exécuter les programmes, vous aurez besoin des packages contenus dans le fichier requirements.txt. Vous pouvez tous les installer avec la commande `pip install -r requirements.txt`

#### INPUT
Le seul input à ce package est le dataset UTKFace.
Il est téléchargeable à l'adresse suivante : `https://drive.google.com/drive/folders/1HROmgviy4jUUUaCdvvrQ8PcqtNg2jn3G?usp=sharing`

### LANCEMENT DES SCRIPTS
Se positionner dans le répertoire et lancer une console avec l'environnement adapté

```
%run get_age_gender_ethnicity.py
```