# NeuroAlign - AI-Powered Fatigue Detection & Smart Scheduling System

## 🧠 Overview

NeuroAlign is a comprehensive fatigue detection system that combines **mental burnout forecasting** with **bio-rhythm based smart scheduling** to proactively monitor and manage cognitive fatigue.

## 🚀 Key Features

### 🎯 Mental Burnout Forecaster
- **Real-time micro-expression analysis** via webcam
- **Typing pattern analysis** (hesitation, backspaces, rhythm)
- **Blink rate monitoring** and facial tension detection
- **Proactive fatigue alerts** before symptoms worsen

### ⏰ Bio-Rhythm Smart Scheduler
- **Wearable device integration** (Fitbit, Garmin, Apple Watch)
- **Energy level prediction** using ML models
- **Intelligent meeting scheduling** based on biological readiness
- **Burnout prevention** through optimal task alignment

## 🛠️ Technology Stack

- **Backend**: Python FastAPI, TensorFlow, OpenCV
- **Frontend**: React, TypeScript, WebRTC
- **Database**: PostgreSQL with TimescaleDB
- **ML Models**: CNN for micro-expressions, LSTM for bio-rhythms
- **Real-time**: WebSocket for live data streaming

## 📁 Project Structure

```
neuroalign/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── models/         # ML models
│   │   ├── api/            # API endpoints
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utilities
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── hooks/          # Custom hooks
│   │   ├── services/       # API calls
│   │   └── utils/          # Utilities
├── ml_models/              # Trained AI models
├── data/                   # Training data
└── docs/                   # Documentation
```

## 🚀 Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd neuroalign
   ```

2. **Backend setup**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

3. **Frontend setup**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🔬 How It Works

### 1. Fatigue Detection Pipeline
1. **Webcam Analysis**: Real-time facial micro-expressions
2. **Typing Dynamics**: Keystroke timing and patterns
3. **Bio-signals**: Heart rate, sleep data from wearables
4. **ML Processing**: Combined analysis for fatigue score
5. **Alert System**: Proactive notifications and recommendations

### 2. Smart Scheduling Algorithm
1. **Data Collection**: Historical energy patterns
2. **ML Prediction**: Energy level forecasting
3. **Schedule Optimization**: Meeting and task alignment
4. **Burnout Prevention**: Low-energy period warnings

## 📊 Data Privacy

- All data processing happens locally when possible
- Biometric data is encrypted and anonymized
- User consent required for data collection
- GDPR compliant data handling

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.