# 🤖 Classification Service

> **AI-Powered Industrial Insurance Fraud Detection**  
> Microservice IA — Model 2 : Failure Classification (XGBoost)  
> Port : **8002**

---

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Dataset](#dataset)
- [Classes prédites](#classes-prédites)
- [Architecture](#architecture)
- [Installation](#installation)
- [Entraînement](#entraînement)
- [API Endpoints](#api-endpoints)
- [Tests](#tests)
- [Docker](#docker)
- [Intégration](#intégration)

---

## 🎯 Vue d'ensemble

Ce microservice analyse les **données capteurs CSV** d'équipements industriels et prédit si une panne déclarée est :
- Une vraie panne mécanique
- Une panne fabriquée (fraude)
- Un sabotage délibéré
- Une usure normale

Il retourne un **score de fraude de 0 à 100** utilisé dans le pipeline de détection.

**Score final du pipeline :**
```
Score = 0.35×anomaly + 0.25×classification + 0.20×NLP + 0.20×vision
```

---

## 📊 Dataset

**Machine Predictive Maintenance Classification** — Kaggle  
`shivamb/machine-predictive-maintenance-classification`

| Colonne | Description |
|---------|-------------|
| `Type` | Type de machine (L/M/H) |
| `Air temperature [K]` | Température ambiante |
| `Process temperature [K]` | Température de la machine |
| `Rotational speed [rpm]` | Vitesse de rotation |
| `Torque [Nm]` | Couple appliqué |
| `Tool wear [min]` | Usure de l'outil |
| `Failure Type` | Type de panne (label) |

---

## 🏷️ Classes prédites

| ID | Classe | Description | Score Fraude |
|----|--------|-------------|--------------|
| 0 | `FAKE` | Panne fabriquée — **FRAUDE** | 90/100 |
| 1 | `NORMAL_WEAR` | Usure normale | 5/100 |
| 2 | `REAL_FAILURE` | Vraie panne mécanique | 30/100 |
| 3 | `SABOTAGE` | Sabotage délibéré — **FRAUDE** | 95/100 |

---

## 🏗️ Architecture

```
classification-service/
├── app/
│   └── main.py                         # FastAPI app (port 8002)
├── models/
│   └── classification/
│       ├── preprocessor.py             # Feature engineering
│       ├── model.py                    # XGBoost training & inference
│       └── artifacts/
│           ├── classifier.pkl          # Modèle entraîné
│           └── feature_importance.png  # Graphique importance
├── tests/
│   └── test_preprocessor.py           # Tests unitaires
├── train_maintenance.py               # Script d'entraînement
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

**Features engineered :**

| Feature | Calcul | Rôle |
|---------|--------|------|
| `Power` | Torque × RPM | Puissance réelle |
| `Temp_diff` | Process_temp - Air_temp | Surchauffe |
| `Wear_rate` | Tool_wear / RPM | Vitesse d'usure |

**Données synthétiques générées :**
- `FAKE` : copie de REAL_FAILURE avec Torque=0, RPM=0, Power=0
- `SABOTAGE` : copie de NORMAL_WEAR avec température×4, Torque×4

---

## ⚙️ Installation

### Prérequis
- Python 3.11+
- pip

### Installation locale

```bash
# Cloner le projet
cd fraud-detection/ai-services

# Créer environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Entraînement

```bash
# Entraîne le modèle depuis Kaggle (téléchargement automatique)
python train_maintenance.py
```

**Résultats attendus :**
```
✅ Accuracy:  99% (requis > 80%)
✅ Precision: 99% (requis > 75%)
✅ Recall:    99% (requis > 80%)
```

---

## 📡 API Endpoints

### Démarrer le serveur

```bash
python main.py
# ou
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

Swagger UI : `http://localhost:8002/docs`

---

### `GET /health`

Vérifie l'état du service.

```bash
curl http://localhost:8002/health
```

```json
{
  "status": "healthy",
  "service": "classification-service",
  "model_loaded": true
}
```

---

### `POST /classify-failure` ⭐ (Endpoint principal)

Prédit la classe depuis un fichier CSV capteurs.

```bash
curl -X POST http://localhost:8002/classify-failure \
  -F "file=@predictive_maintenance.csv"
```

**Réponse :**
```json
{
  "predicted_class": "NORMAL_WEAR",
  "fraud_score": 5,
  "row_count": 10000,
  "class_distribution": {
    "NORMAL_WEAR": 8687,
    "REAL_FAILURE": 1289,
    "FAKE": 23,
    "SABOTAGE": 1
  },
  "feature_importance": {
    "Air_temperature": 0.4146,
    "Power": 0.4131,
    "Rotational_speed": 0.0682,
    "Torque": 0.0673
  },
  "model": "XGBoost",
  "service": "classification-service"
}
```

---

### `POST /classify-features`

Prédit depuis un JSON (une seule lecture capteur).

```bash
curl -X POST http://localhost:8002/classify-features \
  -H "Content-Type: application/json" \
  -d '{
    "Type": "M",
    "Air_temperature": 298.1,
    "Process_temperature": 308.6,
    "Rotational_speed": 0,
    "Torque": 0,
    "Tool_wear": 0
  }'
```

**Réponse :**
```json
{
  "prediction": "FAKE",
  "fraud_score": 90
}
```

---

### Exemples de test

| Scénario | RPM | Torque | Temp | Résultat | Score |
|----------|-----|--------|------|----------|-------|
| Normal | 1551 | 42.8 | 298/308 | NORMAL_WEAR | 5 |
| Vraie panne | 1408 | 68.0 | 298/318 | REAL_FAILURE | 30 |
| Fraude 🚨 | 0 | 0 | 298/308 | FAKE | 90 |
| Sabotage 🚨 | 1400 | 180 | 892/1200 | SABOTAGE | 95 |

---

### `GET /feature-importance`

Retourne le graphique PNG d'importance des features.

```
GET http://localhost:8002/feature-importance
```

---

### `POST /train`

Réentraîne le modèle depuis un CSV uploadé.

```bash
curl -X POST http://localhost:8002/train \
  -F "file=@predictive_maintenance.csv"
```

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ -v --cov=models
```

---

## 🐳 Docker

### Build & Run

```bash
# Build
docker build -t classification-service .

# Run
docker run -p 8002:8002 classification-service
```

### Docker Compose (avec tout le projet)

```bash
docker-compose up -d
```

---

## 🔗 Intégration Backend (NestJS)

Le Backend appelle ce service via BullMQ Worker :

```typescript
// Appel depuis le worker NestJS
const response = await axios.post(
  'http://classification-service:8002/classify-failure',
  formData,
  { headers: { 'Content-Type': 'multipart/form-data' } }
);

const { predicted_class, fraud_score, feature_importance } = response.data;
```

**Score utilisé dans la formule finale :**
```
Score_final = 0.35×S_anomaly + 0.25×fraud_score + 0.20×S_NLP + 0.20×S_vision
```

---

## 👩‍💻 Réalisé par

**KHETIB Asma** — Université M'Hamed Bougara de Boumerdès  
Projet Pluridisciplinaire 2025/2026  
Encadré par **Dr. YAHIATENE Youcef**