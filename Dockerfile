# Image de base : Python léger (version "slim" = sans outils inutiles, pour un conteneur plus petit)
FROM python:3.11-slim

# Dossier de travail à l'intérieur du conteneur
WORKDIR /app

# On copie d'abord uniquement requirements.txt (optimisation : voir explication plus bas)
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# On copie ensuite le reste du code (main.py, dossier models/)
COPY main.py .
COPY models/ ./models/

# Port sur lequel l'API va écouter à l'intérieur du conteneur
EXPOSE 8000

# Commande exécutée au démarrage du conteneur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
