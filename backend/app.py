from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import joblib
import os
import numpy as np

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")
app.config['SECRET_KEY'] = 'signbridge_secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../gesture_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("../frontend", path)

@socketio.on("inference")
def handle_inference(data):
    """
    Expects data to be a list of 63 values (21 landmarks * 3 coords: x, y, z)
    """
    try:
        landmarks = data.get("landmarks")
        if not landmarks or len(landmarks) != 63:
            emit("response", {"error": "Invalid landmarks data"})
            return

        # Predict using the model
        prediction = model.predict([landmarks])[0]
        
        # Send prediction back to client
        emit("response", {"prediction": str(prediction)})
    except Exception as e:
        emit("response", {"error": str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
