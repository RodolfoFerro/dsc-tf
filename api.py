from flask import Flask
from flask import request
from flask import jsonify
from flask import redirect
from flask import url_for
import numpy as np

from source.model_utils import load_model


# Load a pre-trained model form path
model_path = 'models/checkpoint'
model = load_model(model_path)

# Globals for model inference
result_dict = {
    0: 'NEGATIVE',
    1: 'POSITIVE'
}

# API from Flask instance
api = Flask(__name__)


@api.route('/', methods=['GET'])
def home():
    """Redirect route to /status."""

    return redirect(url_for('status'))

@api.route('/api/status', methods=['GET'])
def status():
    """GET method for API status verification."""
    
    message = {
        "status": 200,
        "message": [
            "This API is up and running!"
        ]
    }
    response = jsonify(message)
    response.status_code = 200

    return response


@api.route('/api/predict', methods=['POST'])
def model_inference():
    """POST method for model inference."""

    # Get data as JSON from POST
    data = request.get_json()
    
    # Parse data from JSON
    raw_text = data['input_text']
    
    # Predict using DL model
    raw_prediction = model.predict([raw_text])
    prediction = np.round(raw_prediction)

    
    # Serialize predictions for response
    raw_pred_vector = [float(val) for val in raw_prediction[0]]
    pred_vector = [int(val) for val in prediction[0]]
    
    # Send response
    message = {
        "status": 200,
        "message": [
            {
                "task": "Sentiment Analysis",
                "pred_vector": raw_pred_vector,
                "class_id": pred_vector,
                "class_name": result_dict[pred_vector[0]],
                "input_text": raw_text
            }
        ]
    }
    response = jsonify(message)
    response.status_code = 200

    return response


@api.errorhandler(404)
def not_found(error=None):
    """ GET method for not found routes."""
    
    message = {
        "status": 404,
        "message": [
            "[ERROR] URL not found."
        ]
    }
    response = jsonify(message)
    response.status_code = 404
    
    return response


if __name__ == '__main__':
    api.run(port=8080, debug=True)