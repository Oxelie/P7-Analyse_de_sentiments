# P7-Analyse_de_sentiments

**Introduction**

Ce projet a pour objectif de développer un prototype d'IA capable d'anticiper les bad buzz sur les réseaux sociaux pour la compagnie aérienne Air Paradis. En analysant les tweets, le modèle prédit si la phrase est positive ou négative. Ce projet inclut des méthodologies classiques et avancées, avec une orientation MLOps pour le suivi et le déploiement en production.


🗂 **Structure des Fichiers**

api.py : Code Flask pour exposer le modèle sous forme d'API.

utils/download_model.py : Script pour télécharger les artefacts depuis MLFlow.

tests/test-unit-api.py : Tests unitaires pour vérifier les fonctionnalités de l'API.

DockerfileAPI & docker-compose.yml : Configuration pour la containerisation et le déploiement sur le cloud via unit_test_api.yml.

***notebooks***/ :

1-modele-classique.ipynb : Modèle classique basé sur régression logistique (testés sur 20 000 tweets).

2-modele-avance-BERT.ipynb : Modèle de Word embeddings et LSTM, utilisation également de modèle pré-entrainés (testés sur 20 000 tweets).

3-modele-BERT.ypnb : Fine tuning d'un modèle BERT et essais de fine tuning (testés sur 20 000 tweets).

artifacts/utils/download_model.py : Contient les modèles et vectorizers téléchargés depuis MLFlow à l'aide du script.

***README.md*** : Documentation du projet.

article/README_Analyse_Sentiments.md : Article sur l'évaluation et le résultat de plusieurs modèles d'analyses de sentiments obtenus grâce aux différents notebooks.


🚀 **Technologies Utilisées**

***Frameworks et Bibliothèques :***

Flask : Pour exposer le modèle en API.

TensorFlow/Keras : Développement des modèles avancés.

Scikit-learn : Modèle classique et vectorisation.

MLFlow : Suivi des expérimentations et gestion des artefacts.

***MLOps :***

Azure Application Insights : Suivi des performances en production.

***Containerisation :***

Docker et Docker Compose.


🛠 **Instructions pour Exécuter le Projet**

***Prérequis***

1. Build les images Docker pour le fichier docker-compose.yml :

docker compose build

2. Lancer les images Docker:

docker compose up -d

3. Pour le suivi des expérimentations MLFlow:

http://localhost:5000

***Exécution en Local***

1. API :

Démarrer l'API Flask :

python api.py

Tester une prédiction :

curl -X POST -H "Content-Type: application/json" -d '{"text": "Je suis heureux"}' http://127.0.0.1:8000/predict

2. Tests unitaires :

Exécuter les tests unitaires :

pytest test-unit-api.py

3. Interface :

Interface locale pour tester les prédictions dans un Notebook interface-api.ipynb.


📊 **Approches Modélisées**

***Modèle Classique***

Algorithme : Régression Logistique.

Vectorisation : TF-IDF (utilisé en exemple dans cet API) / CountVectorizer avec lemmatization et stemming.

***Modèles Avancés***

Embeddings avec ['w2v', 'fasttext', 'bert', 'use'] + Régression Logistique

LSTM avec embeddings Word2Vec et FastText.

Différent modèles BERT (pré-entrainé et entrainé) pour un meilleur contexte sémantique.

***Méthodologies MLOps***

1. Tracking avec MLFlow :

Suivi des expérimentations : Accuracy, AUC, hyperparamètres, temps d'entraînement.

Centralisation des artefacts et des modèles.

2. Déploiement :

Pipeline CI/CD avec tests unitaires automatiques et mise en production via GitHub Actions.


🧪 **Tests et Validation**

***Tests unitaires :***

Vérification du chargement du modèle.

Tests de prédictions valides pour des tweets positifs et négatifs.

Code coverage inclus dans le github Actions.

🌐 **Déploiement**

Déploiement via Docker sur le cloud Azure.

Suivi des erreurs et alertes grâce à Azure Application Insights :

Traces des tweets mal classifiés.

Alerte déclenchée après 3 erreurs en moins de 5 minutes.


⚠️ **Disclaimer de Sécurité**

***Limitations Actuelles***

1. Connexion Non Sécurisée :

L'API utilise actuellement HTTP sans chiffrement. Cela signifie que les données transmises entre l'interface et l'API ne sont pas protégées.
Les utilisateurs sont invités à configurer un serveur HTTPS pour garantir la sécurité des échanges.

2. Absence d'Authentification :

Aucune méthode d'authentification (par exemple, token ou clé API) n'est implémentée. Cela rend l'API vulnérable à des usages non autorisés.

3. Sécurité des Données :

Les mécanismes de sécurité comme le chiffrement des données sensibles ne sont pas en place.

***Recommandations pour Améliorer la Sécurité***

1. Mettre en place HTTPS :

Configurer un certificat SSL/TLS pour sécuriser les connexions.

Utiliser des outils comme Let's Encrypt pour un certificat gratuit et fiable.

2. Ajouter un Token d'Authentification :

Implémenter un système de token (par exemple, JWT ou clé API) pour restreindre l'accès.

Configurer des rôles et permissions selon les besoins.

3. Chiffrement des Données :

Chiffrer les données sensibles avant de les transmettre à l'API.

4. Tests de Sécurité :

Effectuer des tests réguliers pour détecter et corriger les vulnérabilités (par exemple, OWASP Top Ten).

**Attention**

L'usage de cette API dans un environnement de production tel quel est fortement déconseillé sans la mise en œuvre des recommandations ci-dessus.


👨‍💻 **Contributeur :**
Stéphanie Duhem
email : steduhem@gmail.com