from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path='modele_couverture_livre.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file:
        try:
            img = image.load_img(file.stream, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32')
            img_array /= 255.0  # Normaliser

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

            return jsonify({'result': float(prediction)})

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'No file provided'}), 400
