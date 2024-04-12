# Utiliser une image de base Python légère
FROM python:3.8-slim

# Définition du répertoire de travail
WORKDIR /model

# Copier le fichier modèle TensorFlow Lite dans le conteneur
COPY modele_couverture_livre.tflite /model/
COPY predict.py /model/

RUN pip install flask tensorflow numpy Pillow

EXPOSE 5001

# La commande par défaut qui garde le conteneur en exécution
CMD ["python", "predict.py"]
