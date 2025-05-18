
from flask import Flask, request, jsonify
import pickle
import os
import logging
from mlflow import sklearn
from opencensus.ext.azure.log_exporter import AzureLogHandler 


# Traceur Azure Application Insights
# Votre clé d’instrumentation
INSTRUMENTATION_KEY = 'c2065e7b-253b-42ba-a950-e66e4af255ba'


# Configuration du logger pour Application Insights
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(AzureLogHandler(connection_string=f'InstrumentationKey={INSTRUMENTATION_KEY}'))

# Fonction pour tester la configuration du logger
def test_logger_configuration(message="Test de configuration du logger"):
    logger.info(message)
    return "Log sent" 

#
# Récupérer le chemin du modèle depuis le répertoire local
artifact_dir = "artifacts/reg_log_TfidfVectorizer_lem"

# Initialiser l'application Flask
app = Flask(__name__)



# Fonction de chargement de la piepeline contenant le modèle et le vectorizer
def load_artifacts():
    """Charge la pipeline complète depuis les fichiers locaux."""
    try:
        logger.info(f"Tentative de chargement de la pipeline depuis : {artifact_dir}")
        loaded_pipeline = sklearn.load_model(artifact_dir)
        logger.info(f"Pipeline chargée avec succès depuis : {artifact_dir}")
        return loaded_pipeline
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la pipeline : {e}", exc_info=True)
        return None

# Charger les artefacts au démarrage de l'application
loaded_pipeline = load_artifacts()


# Définir un point d'entrée pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Requête reçue pour /predict")
    # Vérifier que la pipeline est chargée
    if loaded_pipeline is None:
        logger.error("Pipeline non chargée")
        return jsonify({'error': 'La pipeline n\'a pas pu être chargée pour la prédiction'}), 500

    try:
        # Récupérer les données envoyées dans la requête
        data = request.get_json()
        logger.info(f"Données reçues : {data}")

        # Vérifier que le champ "text" est présent
        if "text" not in data:
            logger.error("Champ 'text' manquant dans la requête")
            return jsonify({'error': 'Le champ "text" est manquant dans la requête'}), 400

        # Récupérer le texte
        text_data = data["text"]

        # Prédire avec la pipeline chargée
        predictions = loaded_pipeline.predict([text_data])
        logger.info(f"Prédictions : {predictions}")

        # Retourner les prédictions en format JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

# Avec cette configuration :

# La route /feedback reçoit un feedback pour chaque prédiction, qu'elle soit correcte ou incorrecte.
# En cas de feedback négatif (non_valide), une trace de niveau warning est envoyée.
# En cas de feedback positif (valide), une trace de niveau info est envoyée.

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if "text" not in data or "prediction" not in data or "feedback" not in data:
        return jsonify({'error': 'Requête invalide'}), 400
    
    tweet_text = data["text"]
    prediction_result = data["prediction"]
    feedback_type = data["feedback"]

    # Enregistrer le feedback dans Application Insights ou un autre système de suivi ici
    if feedback_type == "non_valide":
        logger.warning("Prédiction incorrecte", extra={
            "custom_dimensions": {
                "tweet": tweet_text,
                "prediction": prediction_result
            }
        })
    elif feedback_type == "valide":
        logger.info("Prédiction validée", extra={
            "custom_dimensions": {
                "tweet": tweet_text,
                "prediction": prediction_result
            }
        })

    return jsonify({'status': 'Feedback reçu'})


@app.route('/', methods=['GET'])
def home():
    return "L'API est en cours d'exécution !"
# endpoint de test pour vérifier que l'application fonctionne correctement
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Endpoint de test fonctionnel'})

# Lancer l'application
if __name__ == '__main__':
    # Envoyer un log de test au démarrage pour vérifier la configuration
    # test_logger_configuration("Démarrage de l'application - vérification du logger")
    app.run(host='0.0.0.0', port=8000)



