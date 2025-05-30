# Analyse de Sentiments sur Tweets – Air Paradis

## 🚀 Objectif

Ce projet vise à développer un prototype d’IA capable de prédire le sentiment (positif ou négatif) d’un tweet, pour anticiper les bad buzz sur les réseaux sociaux de la compagnie Air Paradis. Il inclut une API Flask, un pipeline MLOps, des notebooks d’expérimentation, et une interface utilisateur interactive sous Jupyter.

---

## 🗂 Structure du dépôt

- `src/api.py` : API Flask pour exposer le modèle de prédiction.
- `src/interface-api.ipynb` : Interface utilisateur interactive (ipywidgets) pour tester l’API.
- `src/requirements.txt` : Dépendances Python.
- `src/tests/` : Tests unitaires de l’API.
- `artifacts/` : Modèles et vectorizers sauvegardés.
- `expérimentations/` : Notebooks d’expérimentation (BoW, Word2Vec, FastText, BERT…).
- `.github/workflows/ci-cd.yml` : Pipeline CI/CD GitHub Actions pour tests et déploiement Azure.

---

## ⚙️ Installation

1. **Cloner le dépôt :**
   ```sh
   git clone https://github.com/Oxelie/P7-Analyse_de_sentiments
   cd P7-Analyse_de_sentiments
   ```

2. **Installer les dépendances :**
   ```sh
   pip install -r src/requirements.txt
   ```

---

## 🏃‍♂️ Lancer l’API localement

```sh
cd src
python api.py
```
L’API sera accessible sur [http://localhost:8000](http://localhost:8000).

---

## 🧪 Tester l’API

- **Tests unitaires :**
  ```sh
  pytest src/tests/
  ```

- **Tester une prédiction avec curl :**
  ```sh
  curl -X POST -H "Content-Type: application/json" -d '{"text": "Je suis heureux"}' http://localhost:8000/predict
  ```

---

## 💻 Interface utilisateur

Lance le notebook `src/interface-api.ipynb` dans Jupyter. Il permet de :
- Sélectionner un tweet,
- Envoyer le texte à l’API,
- Afficher la prédiction,
- Valider ou non la prédiction (feedback envoyé à l’API).

---

## ☁️ Déploiement Azure

Le pipeline CI/CD (`.github/workflows/ci-cd.yml`) :
- Exécute les tests unitaires à chaque push sur `main`.
- Déploie automatiquement l’API sur Azure Web App si les tests passent.

**Suivi en production :**
- Les feedbacks utilisateurs sont tracés dans Azure Application Insights (logs, alertes).

---

## 📚 Modèles et Expérimentations

- Approche classique : Bag-of-Words + Régression Logistique.
- Approches avancées : Word2Vec, FastText, LSTM, BERT.
- Tracking des expériences et artefacts avec MLflow.

---

## 🔒 Sécurité

- L’API ne gère pas encore l’authentification ni le chiffrement HTTPS (à ajouter pour la production).
- Voir la section "Disclaimer de Sécurité" dans le code pour recommandations.

---

## 👩‍💻 Auteur

Stéphanie Duhem  
[steduhem@gmail.com](mailto:steduhem@gmail.com)