from flask import Flask, request, render_template, jsonify
import requests
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Chemin pour enregistrer les fichiers temporaires
UPLOAD_FOLDER = 'temp_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# S'assurer que le dossier d'upload existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Envoyer l'image au service de modèle pour la prédiction
            try:
                with open(filepath, 'rb') as f:
                    files = {'file': (filename, f, 'image/jpeg')}
                    response = requests.post("http://model:5001/predict", files=files)

                # Gérer la réponse
                if response.status_code == 200:
                    prediction_result = response.json().get('result')
                    result = "Ce livre est prédit comme un bon livre." if prediction_result > 0.5 else "Ce livre est prédit comme un mauvais livre."
                else:
                    result = f"Erreur lors de la prédiction: {response.text}"
            except Exception as e:
                result = f"Erreur de communication avec le service de modèle: {str(e)}"
            finally:
                # Supprimer le fichier temporaire
                os.remove(filepath)

            return render_template('result.html', result=result)
        else:
            return render_template('index.html', error="Aucun fichier fourni.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
