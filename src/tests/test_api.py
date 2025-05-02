import unittest
import json
from src.api import app
from unittest.mock import patch

class TestAPI(unittest.TestCase):
    def setUp(self):
        # Configure the Flask test client
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_missing_model(self):
        # Test the case where the model is not loaded
        response = self.app.post('/predict', data=json.dumps({"text": "Test tweet"}), content_type='application/json')
        self.assertEqual(response.status_code, 500)
        self.assertIn('Le modèle ou le vectorizer n\'a pas pu être chargé', response.get_json().get('error'))

    def test_predict_missing_text_field(self):
        # Test the case where the "text" field is missing in the request
        response = self.app.post('/predict', data=json.dumps({}), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('Le champ "text" est manquant dans la requête', response.get_json().get('error'))

    def test_predict_success(self):
        # Mock the loaded_model and tfidf_vectorizer for a successful prediction
        with patch('src.api.loaded_model') as mock_model:
            mock_model.predict.return_value = [1]  # Mock prediction result
            response = self.app.post('/predict', data=json.dumps({"text": "Test tweet"}), content_type='application/json')
            self.assertEqual(response.status_code, 200)
            self.assertIn('predictions', response.get_json())
            self.assertEqual(response.get_json().get('predictions'), [1])

if __name__ == '__main__':
    unittest.main()