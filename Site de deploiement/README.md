# Site Web de Déploiement RLT

Site web Flask pour comparer les performances des modèles RLT avec Random Forest, Gradient Boosting, Lasso/Ridge et ExtraTrees.

## 🎯 Fonctionnalités

- **Sélection de dataset** : Choisir parmi 10 datasets UCI
- **Saisie des features** : Formulaire dynamique adapté à chaque dataset
- **Remplissage aléatoire intelligent** : Génération de valeurs cohérentes basée sur les distributions réelles
- **Comparaison de modèles** : Prédictions côte à côte avec 5 modèles différents
- **Performances historiques** : Métriques de test affichées pour chaque modèle
- **Visualisations** : Graphiques de comparaison interactifs

## 📦 Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Vérifier que les fichiers nécessaires sont présents :
- `../models/` : Dossier contenant tous les modèles .pkl
- `../datasets_metadata.json`
- `../models_performance.json`
- `../feature_ranges.json`

## 🚀 Lancement

```bash
python app.py
```

Le serveur démarre sur `http://localhost:5000`

## 📁 Structure du Projet

```
Site de deploiement/
├── app.py                  # Serveur Flask
├── requirements.txt        # Dépendances Python
├── templates/
│   └── index.html         # Page principale
├── static/
│   ├── css/
│   │   └── style.css      # Styles CSS
│   └── js/
│       └── main.js        # JavaScript client
```

## 🔌 API Endpoints

- `GET /api/datasets` : Liste des datasets disponibles
- `GET /api/dataset/<name>/info` : Informations détaillées d'un dataset
- `POST /api/dataset/<name>/random-fill` : Génère des valeurs aléatoires
- `POST /api/predict` : Effectue des prédictions avec tous les modèles

## 🎨 Interface Utilisateur

### Étape 1 : Sélection du Dataset
Cliquez sur une carte de dataset pour le sélectionner. Les informations (nombre d'échantillons, features, type de tâche) s'affichent automatiquement.

### Étape 2 : Saisie des Features
Deux options :
- **Saisie manuelle** : Remplir chaque champ individuellement
- **Remplissage aléatoire** : Cliquer sur "🎲 Remplissage Aléatoire Intelligent" pour générer des valeurs cohérentes

### Étape 3 : Résultats
Après soumission, les prédictions de tous les modèles s'affichent avec :
- La valeur/classe prédite
- Les performances historiques (Accuracy, MSE, R²)
- Un graphique de comparaison
- Indication du meilleur modèle 🏆

## 🧪 Modèles Comparés

1. **RLT** (Reinforcement Learning Trees) - Meilleure config depuis la simulation
2. **Random Forest** - Forêt aléatoire standard
3. **Gradient Boosting** - Boosting par gradient
4. **ExtraTrees** - Extremely Randomized Trees
5. **Lasso** (régression) ou **Logistic Ridge** (classification)

## 📊 Datasets Disponibles

- Breast Cancer (Classification)
- Boston Housing (Régression)
- Parkinson (Classification)
- Sonar (Classification)
- White Wine Quality (Régression)
- Red Wine Quality (Régression)
- Parkinson Oxford (Régression)
- Ozone (Régression)
- Concrete Strength (Régression)
- Auto MPG (Régression)

## 🛠️ Technologies Utilisées

- **Backend** : Flask (Python)
- **Frontend** : HTML5, CSS3, JavaScript (Vanilla)
- **Machine Learning** : scikit-learn, modèles RLT custom
- **Data** : NumPy, Pandas

## 📝 Notes pour Chercheurs

- Les valeurs générées par remplissage aléatoire respectent les distributions observées dans les données d'entraînement
- Les performances affichées correspondent aux métriques de test calculées lors de l'entraînement
- Le "meilleur modèle" est déterminé par la plus haute accuracy (classification) ou la prédiction la plus proche de la moyenne (régression)

## 🐛 Dépannage

**Erreur "Dataset not found"** : Vérifier que le dossier `models/` contient bien tous les sous-dossiers des datasets

**Erreur lors de la prédiction** : Vérifier que tous les fichiers .pkl (modèles + scaler) sont présents dans le dossier du dataset

**Port 5000 déjà utilisé** : Modifier le port dans `app.py` ligne finale : `app.run(debug=True, port=VOTRE_PORT)`
