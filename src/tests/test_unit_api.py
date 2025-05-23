

import pytest
import numpy as np
from flask import Flask
from unittest.mock import MagicMock
from src.api import app, loaded_pipeline, load_artifacts # Importer les objets directement

# Utiliser le client de test de Flask pour simuler des requêtes à l'API
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Tests pour vérifier que la piepline contenant le modèle est bien chargée
def test_load_artifacts_success(monkeypatch):
    # Mock le chargement du modèle pour simuler un succès
    def mock_load_model(path):
        return "mock_pipeline"
    monkeypatch.setattr('src.api.sklearn.load_model', mock_load_model)
    pipeline = load_artifacts()
    assert pipeline == "mock_pipeline"

def test_load_artifacts_failure(monkeypatch):
    # Mock le chargement du modèle pour simuler un échec
    def mock_load_model(path):
        raise Exception("Erreur simulée")
    monkeypatch.setattr('src.api.sklearn.load_model', mock_load_model)
    pipeline = load_artifacts()
    assert pipeline is None
    
# Test pour vérifier que l'API retourne une réponse correcte pour le point d'entrée racine
def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert "L'API est en cours d'exécution".encode('utf-8') in response.data

# Test pour vérifier que l'API retourne une prédiction valide pour un texte positif
def test_predict_positive(client, monkeypatch):
    monkeypatch.setattr('src.api.loaded_pipeline', MagicMock(predict=lambda x: np.array([1])))
    response = client.post('/predict', json={'text': 'Je suis tellement heureux aujourd\'hui!'})
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'predictions' in json_data
    assert isinstance(json_data['predictions'], list)  # Vérifie que les prédictions sont sous forme de liste

# Test pour vérifier que l'API retourne une prédiction valide pour un texte négatif
def test_predict_negative(client, monkeypatch):
    monkeypatch.setattr('src.api.loaded_pipeline', MagicMock(predict=lambda x: np.array([0])))
    response = client.post('/predict', json={'text': 'Je suis très triste et déçu.'})
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'predictions' in json_data
    assert isinstance(json_data['predictions'], list)

# Test pour vérifier que l'API retourne une erreur si le champ "text" est manquant
def test_missing_text_field(client, monkeypatch):
    monkeypatch.setattr('src.api.loaded_pipeline', MagicMock(predict=lambda x: [1]))
    response = client.post('/predict', json={})
    json_data = response.get_json()
    assert response.status_code == 400
    assert 'error' in json_data
    assert json_data['error'] == 'Le champ "text" est manquant dans la requête'

# Test pour vérifier le comportement si le modèle ou le vectorizer ne sont pas chargés
def test_predict_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr('src.api.loaded_pipeline', None)
    response = client.post('/predict', json={'text': 'Test de texte'})
    json_data = response.get_json()
    assert response.status_code == 500
    assert 'error' in json_data
    assert json_data['error'] == "La pipeline n'a pas pu être chargée pour la prédiction"

# Test pour vérifier que l'API gère correctement une exception pendant la prédiction
def test_predict_exception(client, monkeypatch):
    # On mock la méthode predict de la pipeline pour simuler une erreur
    def mock_predict(data):
        raise Exception("Erreur dans predict")

    monkeypatch.setattr('src.api.loaded_pipeline', MagicMock())
    monkeypatch.setattr('src.api.loaded_pipeline.predict', mock_predict)
    response = client.post('/predict', json={'text': 'Test de texte'})
    json_data = response.get_json()
    assert response.status_code == 400
    assert 'error' in json_data
    assert json_data['error'] == "Erreur dans predict"

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
    monkeypatch.setattr('src.api.logger', mock_logger)

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
    monkeypatch.setattr('src.api.logger', mock_logger)

    client.post('/feedback', json={
        "text": "Ceci est un texte",
        "prediction": "positif",
        "feedback": "valide"
    })

    mock_logger.info.assert_called_once_with(
        "Prédiction validée",
        extra={"custom_dimensions": {"tweet": "Ceci est un texte", "prediction": "positif"}}
    )
