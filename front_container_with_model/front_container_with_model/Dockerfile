
# Utiliser une image de base qui inclut Python.
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur.
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur.
COPY main.py /app/
COPY static /app/static
COPY templates /app/templates
COPY requirements.txt /app/

# Installer Flask, TensorFlow Lite et autres dépendances nécessaires.
RUN pip install Flask tensorflow numpy Pillow

# Exposer le port utilisé par l'application Flask.
EXPOSE 5000

# Définir la variable d'environnement pour Flask
ENV FLASK_APP=main.py

# Commande pour démarrer l'application Flask.
CMD ["flask", "run", "--host=0.0.0.0"]
