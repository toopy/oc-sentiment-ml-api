# De l'entraînement à la supervision: un pipeline MLOps complet pour la classification de sentiments

## Introduction

Dans le cadre d'un projet de classification de sentiments à partir de tweets, nous avons exploré et comparé trois approches de modélisation, avant de les intégrer dans une démarche MLOps complète.

Cette démarche inclut la création de modèles, leur déploiement avec Docker sur Azure, le suivi des performances en production via Application Insights, ainsi que la gestion du cycle de vie du modèle.

## Partie 1 — Trois approches de modélisation: synthèse et comparaison

Dans cette première phase du projet, plusieurs approches de modélisation ont été testées, comparées et évaluées grâce à l'outil de suivi d'expérience MLflow. Ce tableau de bord nous a permis de suivre automatiquement la performance de chaque modèle (accuracy, F1-score, RMSE, etc.).

<img src="./docs/img/Capture%20d%E2%80%99%C3%A9cran%20du%202025-06-16%2018-14-38.png" alt="Interface MLflow affichant les performances comparées des modèles de classification de sentiments" width="700"/>

_Figure – Interface MLflow utilisée pour visualiser les performances comparées des différents modèles testés._

### 1. Modèle sur mesure simple

- **Structure**: pipeline Scikit-learn avec TfidfVectorizer et LogisticRegression
- **Entraînement**: rapide, peu coûteux, avec recherche d'hyperparamètres par grille (GridSearchCV)
- **Points forts**: pipeline simple à déployer, bon compromis performance/complexité
- **Limites**: performances limitées face aux structures linguistiques complexes

### 2. Modèle sur mesure avancé

- **Structure**: Tokenizer Keras + Embedding + GlobalAveragePooling + Dense (softmax)
- **Entraînement**: rapide, peu coûteux
- **Points forts**: facile à entraîner, robuste avec peu de données
- **Limites**: performances réduites, difficulté à généraliser

### 3. Modèle BERT fine-tuné

- **Structure**: `bert-base-uncased` via HuggingFace Transformers + couche de classification
- **Points forts**: performance élevée, compréhension fine du langage naturel
- **Limites**: modèle lourd, entraînement sur GPU recommandé, déploiement plus complexe

### Évaluation des modèles avec MLflow

Afin de comparer objectivement les modèles testés, nous avons utilisé MLflow pour enregistrer et suivre leurs performances. Chaque expérience enregistrée comprend les paramètres d'entraînement, les hyperparamètres testés, ainsi que plusieurs métriques clés :

- **Accuracy**: proportion globale de prédictions correctes.
- **Precision**: capacité du modèle à éviter les faux positifs (important si on veut éviter de surclasser une opinion positive par erreur).
- **Recall**: capacité à détecter toutes les vraies occurrences d’un sentiment (éviter les faux négatifs).
- **F1-score**: moyenne harmonique entre précision et rappel, idéale en cas de classes déséquilibrées.

Nous avons utilisé principalement le F1-score comme critère de sélection du meilleur modèle car les classes (positif, négatif, neutre) sont souvent inégalement représentées dans des jeux de tweets réels.

Chaque exécution dans MLflow a permis de tracer l’évolution de ces métriques en fonction des hyperparamètres. Cela nous a permis de retenir :

- pour le modèle simple, un `TfidfVectorizer` avec n-gram (1,2) et `LogisticRegression` avec `C=1` et `penalty=l2`.
- pour le modèle avancé, un `LSTM` avec embedding_dim=128 et dropout=0.5.
- pour le modèle BERT, un fine-tuning sur 3 époques avec un taux d’apprentissage de `2e-5`.
- etc.

### Synthèse comparative

| Critère                 | Modèle simple | Modèle avancé | BERT fine-tuné |
|-------------------------|---------------|---------------|----------------|
| Facilité d'entraînement | ✨✨✨✨          | ✨✨✨           | ✨✨             |
| Temps d'entraînement    | ✨✨✨           | ✨✨✨✨          | ✨              |
| Performance             | ✨✨            | ✨✨✨✨          | ✨✨✨✨✨          |

## Partie 2 — Une démarche MLOps concrète

### Principes clés du MLOps appliqués ici

- **Traçabilité**: gestion du code via Git/GitHub, versionnement des modèles
- **Automatisation**: tests, build, push Docker, déploiement avec GitHub Actions
- **Surveillance**: Application Insights pour logs et alertes
- **Cycle de vie complet**: collecte de feedback utilisateur, suivi d’erreurs, amélioration continue

### Étapes mises en œuvre

#### 1. Tracking et versioning

- Code versionné avec Git
- Modèles sérialisés avec `joblib`, stockés dans Azure Blob
- GitHub Actions gère le tagging d’images Docker avec le hash de commit

<img src="./docs/img/Capture%20d%E2%80%99%C3%A9cran%20du%202025-06-16%2016-17-11.png" alt="Dépôt GitHub avec branche first-implementation" width="700"/>

_Figure – Dépôt public avec pipeline CI/CD complet pour une API de prédiction de sentiments._

#### 2. Tests unitaires

- Tests `pytest` sur les endpoints de l'API FastAPI
- Cas testés: texte normal, vide, avec caractères spéciaux, erreurs

#### 3. Déploiement

- Dockerfile exposant le port 8000
- GitHub Actions: build de l’image, push sur Docker Hub, déploiement sur Azure Container Apps (conteneur Linux)
- Variable `WEBSITES_PORT=8000` définie dans la configuration Azure

<img src="./docs/img/Capture%20d%E2%80%99%C3%A9cran%20du%202025-06-16%2016-17-31.png" alt="Pipeline GitHub Actions en cours" width="700"/>

_Figure – Vue du workflow GitHub Actions exécutant le déploiement automatique._

## Partie 3 — Suivi de la performance du modèle en production

### Traces dans Application Insights

- SDK `opencensus-ext-azure` intégré dans l’API
- Si un utilisateur signale une erreur de prédiction, la trace est enregistrée:

```python
logger.warning("Bad prediction", extra={
    "custom_dimensions": {
        "tweet_text": text,
        "predicted_label": prediction
    }
})
```

### Requête Kusto typique pour l’analyse

```kusto
traces
| where message == "Bad prediction"
| project timestamp, customDimensions.tweet_text, customDimensions.predicted_label
| order by timestamp desc
```

<img src="./docs/img/Capture%20d%E2%80%99%C3%A9cran%20du%202025-06-16%2016-44-29.png" alt="Traces Application Insights - Requête Kusto" width="700"/>

_Figure – Journal d’erreurs collectées avec Application Insights (KQL)._

### Mise en place d’une alerte automatique

- Alerte déclenchée si ≥ 3 erreurs dans un intervalle de 5 minutes
- Action group: envoi de mail ou SMS
- Créée via Azure Monitor à partir de la requête Kusto

<img src="./docs/img/Capture%20d%E2%80%99%C3%A9cran%20du%202025-06-16%2020-55-05.png" alt="Alerte déclenchée pour trop de tweets mal prédits" width="700"/>

_Figure – Historique d’une alerte déclenchée dans Azure Monitor suite à un nombre élevé de tweets mal classés._

## Partie 4 — Analyse des logs et amélioration continue

<img src="./docs/img/Capture%20d%E2%80%99%C3%A9cran%20du%202025-06-16%2016-45-48.png" alt="Interface Streamlit - Analyse de sentiment" width="700"/>

_Figure – Interface utilisateur pour la détection de sentiment avec retour utilisateur._

### 1. Analyse périodique

- Export des traces vers un `pandas.DataFrame`
- Colonnes: `timestamp`, `tweet_text`, `predicted_label`
- Possibilité de filtrer, grouper, trier, visualiser

<img src="./docs/img/Capture%20d%E2%80%99%C3%A9cran%20du%202025-06-16%2017-54-51.png" alt="Extraction des logs Application Insights dans un DataFrame" width="700"/>

_Figure – Exécution d’une requête Kusto et transformation des résultats en DataFrame Pandas._

### 2. Boucle de réentraînement

La mise en place d’une boucle de réentraînement permet d’assurer que le modèle reste performant face à l’évolution naturelle du langage (nouvelles expressions, hashtags, tournures ironiques). Voici la stratégie déployée:

- **Collecte des erreurs:** les prédictions signalées comme erronées via `/feedback` sont automatiquement loguées dans Application Insights avec le texte et la prédiction du modèle.
- **Extraction et validation:** chaque semaine, les données sont extraites via une requête Kusto et chargées dans un DataFrame. Un annotateur peut alors valider manuellement le vrai label de chaque tweet mal classé.
- **Versioning du dataset:** _(TODO)_ les tweets validés peuvent être ajoutés à un corpus (ex.: feedback_validated.csv), versionné dans Git.
- **Réentraînement:** _(TODO)_ un script de réentraînement recharge le jeu complet enrichi, puis effectue un nouvel entraînement. Un rapport `mlflow` est généré pour vérifier si le nouveau modèle dépasse le précédent.
- **Déploiement continu:**  si les résultats sont meilleurs, le nouveau modèle est versionné dans le dépôt Git et automatiquement intégré dans l’image Docker via GitHub Actions.

### 3. Tableau de bord de supervision (optionnel)

- Intégration des logs Application Insights avec Power BI ou Azure Workbook
- Visualisation du nombre de requêtes, du temps de réponse, et des dernières erreurs de prédiction

<img src="./docs/img/Capture%20d%E2%80%99%C3%A9cran%20du%202025-06-16%2022-14-46.png" alt="Dashboard Azure - Monitoring de l'API et erreurs" width="700"/>

_Figure – Tableau de bord en temps réel pour le suivi des requêtes et des prédictions erronées._

## Conclusion

De la modélisation à la supervision, nous avons mis en place une chaîne MLOps robuste, réplicable et maintenable. Le déploiement via GitHub Actions et Azure Container Apps garantit des mises à jour fluides, tandis que la collecte de feedback permet une amélioration continue basée sur des données réelles.

Cette approche illustre comment combiner des modèles de complexité variable avec des principes de production moderne pour construire un service de classification intelligent, durable, et centré sur l’utilisateur.
