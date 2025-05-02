
# version johnathan doit corrspondre aux fonctions de api.py:

import pytest
from flask import Flask
from unittest.mock import MagicMock
from api import app, loaded_model, tfidf_vectorizer, load_artifacts  # Importer les objets directement

# Utiliser le client de test de Flask pour simuler des requêtes à l'API
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Test pour vérifier que le modèle et le vectorizer sont bien chargés
def test_model_and_vectorizer_loaded():
    assert loaded_model is not None, "Le modèle n'a pas été chargé correctement dans l'API."
    assert tfidf_vectorizer is not None, "Le vectorizer n'a pas été chargé correctement dans l'API."

# Test pour vérifier que l'API retourne une prédiction valide pour un texte positif
def test_predict_positive(client):
    response = client.post('/predict', json={'text': 'Je suis tellement heureux aujourd\'hui!'})
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'predictions' in json_data
    assert isinstance(json_data['predictions'], list)  # Vérifie que les prédictions sont sous forme de liste

# Test pour vérifier que l'API retourne une prédiction valide pour un texte négatif
def test_predict_negative(client):
    response = client.post('/predict', json={'text': 'Je suis très triste et déçu.'})
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'predictions' in json_data
    assert isinstance(json_data['predictions'], list)

# Test pour vérifier que l'API retourne une erreur si le champ "text" est manquant
def test_missing_text_field(client):
    response = client.post('/predict', json={})
    json_data = response.get_json()
    assert response.status_code == 400
    assert 'error' in json_data
    assert json_data['error'] == 'Le champ "text" est manquant dans la requête'

# Test pour vérifier le comportement si le modèle ou le vectorizer ne sont pas chargés
def test_predict_model_not_loaded(client, monkeypatch):
    # Forcer loaded_model à None
    monkeypatch.setattr('api.loaded_model', None)
    response = client.post('/predict', json={'text': 'Test de texte'})
    json_data = response.get_json()
    assert response.status_code == 500
    assert 'error' in json_data
    assert json_data['error'] == "Le modèle ou le vectorizer n'a pas pu être chargé pour la prédiction"

# Test pour vérifier que l'API gère correctement une exception pendant la prédiction
def test_predict_exception(client, monkeypatch):
    def mock_transform(data):
        raise Exception("Erreur dans transform")

    monkeypatch.setattr('api.tfidf_vectorizer.transform', mock_transform)
    response = client.post('/predict', json={'text': 'Test de texte'})
    json_data = response.get_json()
    assert response.status_code == 400
    assert 'error' in json_data
    assert json_data['error'] == "Erreur dans transform"

# Test pour le chargement des artefacts - succès
def test_load_artifacts_success(monkeypatch):
    def mock_load_model(path):
        return "mock_model"

    def mock_open(path, mode):
        return MagicMock()

    monkeypatch.setattr('api.sklearn.load_model', mock_load_model)
    monkeypatch.setattr('builtins.open', mock_open)
    monkeypatch.setattr('pickle.load', lambda f: "mock_vectorizer")

    model, vectorizer = load_artifacts()
    assert model == "mock_model"
    assert vectorizer == "mock_vectorizer"

# Test pour le chargement des artefacts - échec
def test_load_artifacts_failure(monkeypatch):
    def mock_load_model(path):
        raise Exception("Erreur simulée")

    monkeypatch.setattr('api.sklearn.load_model', mock_load_model)
    model, vectorizer = load_artifacts()
    assert model is None
    assert vectorizer is None

# Test pour le point d'entrée feedback - cas valide
def test_feedback_valid(client):
    response = client.post('/feedback', json={
        "text": "Ceci est un texte",
        "prediction": "positif",
        "feedback": "valide"
    })
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'status' in json_data
    assert json_data['status'] == 'Feedback reçu'

# Test pour le point d'entrée feedback - cas de requête invalide
def test_feedback_invalid_request(client):
    response = client.post('/feedback', json={})
    json_data = response.get_json()
    assert response.status_code == 400
    assert 'error' in json_data
    assert json_data['error'] == 'Requête invalide'

# Test pour vérifier les logs dans le feedback - non valide
def test_feedback_log_non_valide(client, monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr('api.logger', mock_logger)

    client.post('/feedback', json={
        "text": "Ceci est un texte",
        "prediction": "positif",
        "feedback": "non_valide"
    })

    mock_logger.warning.assert_called_once_with(
        "Prédiction incorrecte",
        extra={"custom_dimensions": {"tweet": "Ceci est un texte", "prediction": "positif"}}
    )

# Test pour vérifier les logs dans le feedback - valide
def test_feedback_log_valide(client, monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr('api.logger', mock_logger)

    client.post('/feedback', json={
        "text": "Ceci est un texte",
        "prediction": "positif",
        "feedback": "valide"
    })

    mock_logger.info.assert_called_once_with(
        "Prédiction validée",
        extra={"custom_dimensions": {"tweet": "Ceci est un texte", "prediction": "positif"}}
    )
