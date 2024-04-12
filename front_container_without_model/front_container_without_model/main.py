from flask import Flask, request, render_template, jsonify, redirect, url_for
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Vérifiez si le fichier a été envoyé
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # Si l'utilisateur ne sélectionne pas de fichier, le navigateur envoie
    # un fichier sans nom.
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        # Vous pouvez sauvegarder le fichier quelque part si nécessaire
        # file.save(os.path.join('/path/to/the/uploads', filename))
        try:
            # Envoi du fichier au service de modèle
            response = requests.post("http://model:5001/predict", files={'file': file})
            if response.status_code == 200:
                result = response.json()
                return jsonify(result)
            else:
                return jsonify({'error': 'Failed to get prediction from the model service'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'No file provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
