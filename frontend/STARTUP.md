# 🚀 Frontend Startup & Complete Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Project Structure](#project-structure)
5. [Using the Frontend](#using-the-frontend)
6. [API Documentation](#api-documentation)
7. [Troubleshooting](#troubleshooting)
8. [Production Deployment](#production-deployment)

---

## ⚡ Quick Start

**Total time: 10 minutes**

### One-Command Setup (Windows PowerShell)

```powershell
# Terminal 1: Frontend
cd "d:\College\Final year Project\SChypergraph\frontend"
npm install
npm run dev

# Terminal 2: Backend
cd "d:\College\Final year Project\SChypergraph"
pip install fastapi uvicorn torch
python frontend/backend_server.py

# Terminal 3: Open browser
start http://localhost:5173
```

### One-Command Setup (Linux/Mac)

```bash
# Terminal 1: Frontend
cd frontend
npm install && npm run dev

# Terminal 2: Backend
pip install fastapi uvicorn torch
python frontend/backend_server.py

# Terminal 3: Open browser
open http://localhost:5173
```

---

## 📦 Installation

### Prerequisites Check

```bash
# Check Node.js
node --version          # Should be 18+
npm --version          # Should be 9+

# Check Python
python --version       # Should be 3.8+
```

### Step 1: Install Frontend Dependencies

```bash
cd frontend
npm install
```

This installs:
- ✅ React 18.2.0
- ✅ TypeScript 5.2.2
- ✅ Vite 4.5.0
- ✅ Tailwind CSS
- ✅ Recharts
- ✅ Axios
- ✅ Lucide React

### Step 2: Install Backend Dependencies

```bash
pip install fastapi uvicorn torch
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Frontend
npm --version

# Backend
python -m pip list | findstr fastapi  # Windows
python -m pip list | grep fastapi     # Linux/Mac
```

---

## 🏃 Running the Application

### Method 1: Using npm Scripts (Recommended)

**Terminal 1 - Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 2 - Backend:**
```bash
cd d:\College\Final year Project\SChypergraph
python frontend/backend_server.py
```

**Output:**
```
Frontend: Local: http://localhost:5173/
Backend:  Uvicorn running on http://0.0.0.0:8000
```

### Method 2: Using npm & Python Directly

**Terminal 1:**
```bash
cd frontend
npm run dev
```

**Terminal 2:**
```bash
cd frontend
python backend_server.py
```

### Method 3: Production Build

```bash
# Build optimized bundle
cd frontend
npm run build

# Check dist/ folder was created
ls dist/

# Serve production build
npm run preview
```

### Method 4: With Docker (Optional)

Create `Dockerfile` in frontend/:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview"]
```

Build and run:
```bash
docker build -t ht-hgnn-frontend .
docker run -p 5173:5173 ht-hgnn-frontend
```

---

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.tsx              # Navigation & routing
│   │   │   └── Pages: home, analysis, model, about
│   │   │
│   │   ├── FileUpload.tsx          # CSV uploader
│   │   │   ├── Drag & drop support
│   │   │   ├── File validation
│   │   │   └── Progress indicator
│   │   │
│   │   ├── RiskDashboard.tsx       # Risk visualization
│   │   │   ├── Summary cards (total nodes, avg risk, critical count)
│   │   │   ├── Criticality bar chart
│   │   │   ├── Risk distribution line chart
│   │   │   └── Top critical nodes list
│   │   │
│   │   └── ModelStats.tsx          # Performance metrics
│   │       ├── Loss convergence chart
│   │       ├── Training metrics (initial/final loss)
│   │       └── Device & memory info
│   │
│   ├── services/
│   │   └── inferenceService.ts    # API client wrapper
│   │       ├── getModelInfo()
│   │       ├── predict()
│   │       ├── analyze()
│   │       ├── uploadAndPredict()
│   │       ├── getHealth()
│   │       └── getTrainingHistory()
│   │
│   ├── types/
│   │   └── index.ts               # TypeScript interfaces
│   │       ├── PredictionInput
│   │       ├── PredictionOutput
│   │       ├── RiskAssessment
│   │       ├── AnalysisResult
│   │       ├── TrainingMetrics
│   │       └── ModelInfo
│   │
│   ├── App.tsx                     # Main component & routing
│   ├── main.tsx                    # React entry point
│   └── index.css                   # Global Tailwind styles
│
├── public/                         # Static assets
├── index.html                      # HTML template
├── vite.config.ts                  # Vite bundler config
├── tailwind.config.js              # Tailwind CSS config
├── postcss.config.js               # PostCSS config
├── tsconfig.json                   # TypeScript config
├── tsconfig.node.json              # TypeScript Node config
├── package.json                    # Dependencies & scripts
│
├── backend_server.py               # FastAPI backend
├── requirements.txt                # Python dependencies
│
├── README.md                       # Project overview
├── SETUP_GUIDE.md                  # Installation guide
├── .gitignore                      # Git ignore rules
├── .env.example                    # Environment variables template
└── STARTUP.md                      # This file
```

---

## 🎯 Using the Frontend

### Page 1: Dashboard (Home)

**Location**: `http://localhost:5173/`

**Features**:
1. **Header** with logo and navigation
2. **Model Information Card**:
   - Model name
   - Total parameters
   - Device (CUDA/CPU)
   - Training date
3. **File Uploader**:
   - Drag & drop CSV files
   - Click to browse
   - Real-time validation
4. **Upload Success**:
   - Redirects to Analysis page
   - Displays prediction results

**How to use**:
1. Open the page
2. Prepare CSV file with 18 numerical features per row
3. Drag file to upload area OR click to browse
4. Wait for processing
5. View results on Analysis page

### Page 2: Analysis

**Location**: Automatically loaded after file upload

**Features**:
1. **Summary Cards**:
   - Total Nodes Analyzed
   - Average Risk Score
   - Critical Nodes Count
   - High Risk Count

2. **Charts**:
   - Top 10 Criticality Scores (bar chart)
   - Risk Distribution (line chart)

3. **Critical Nodes Table**:
   - Rank
   - Node ID
   - Risk Level (LOW/MEDIUM/HIGH/CRITICAL)
   - Criticality Score
   - Price Prediction
   - Change Forecast
   - Recommendations

**How to interpret**:
- 🟢 GREEN (LOW): < 25% risk → safe supplier
- 🟡 YELLOW (MEDIUM): 25-50% risk → monitor
- 🟠 ORANGE (HIGH): 50-75% risk → mitigate
- 🔴 RED (CRITICAL): > 75% risk → urgent action

### Page 3: Model Performance

**Location**: http://localhost:5173/#model

**Features**:
1. **Key Metrics**:
   - Initial Loss: Starting training loss
   - Final Loss: After 50 epochs
   - Improvement: Percentage improvement
   - Training Time: Total training duration

2. **Details**:
   - Epochs trained
   - Memory used
   - Device used

3. **Loss Convergence Chart**:
   - Initial vs Final loss comparison
   - Visual improvement representation

### Page 4: About

**Location**: http://localhost:5173/#about

**Contains**:
- Project description
- Architecture explanation
- Key features
- Use cases
- Technology stack

---

## 📡 API Documentation

### Backend API Base URL
```
http://localhost:8000
```

### Available Endpoints

#### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-16T15:58:00.000Z"
}
```

#### 2. Get Model Info
```http
GET /model/info
```

**Response**:
```json
{
  "name": "HT-HGNN",
  "version": "1.0.0",
  "parameters": 130445,
  "device": "cuda",
  "architecture": "Heterogeneous Temporal Hypergraph NN",
  "trainingDate": "2026-01-16T15:58:00Z",
  "metrics": {
    "initialLoss": 10375.91,
    "finalLoss": 9055.48,
    "improvement": 1320.43,
    "epochs": 50,
    "totalTime": 75.0,
    "memoryUsage": 2.1
  }
}
```

#### 3. Make Predictions
```http
POST /predict
Content-Type: application/json

{
  "nodeFeatures": [[1, 2, 3, ..., 18], ...],
  "incidenceMatrix": [[0, 1, ...], ...]
}
```

**Response**:
```json
{
  "pricePredictions": [500.5, 502.3, ...],
  "changePredictions": [0.05, 0.03, ...],
  "criticalityScores": [0.75, 0.82, ...],
  "nodeIds": ["node_0", "node_1", ...]
}
```

#### 4. Upload & Predict
```http
POST /upload/predict
Content-Type: multipart/form-data

file: <csv_file>
```

**CSV Format**:
```csv
feature_1,feature_2,...,feature_18
100,200,...,400
150,250,...,450
```

**Response**: Same as /predict

#### 5. Analyze Supply Chain
```http
POST /analyze
Content-Type: application/json

{
  "nodeFeatures": [[...], ...],
  "incidenceMatrix": [[...], ...]
}
```

**Response**:
```json
{
  "timestamp": "2026-01-16T15:58:00Z",
  "predictions": {
    "pricePredictions": [...],
    "changePredictions": [...],
    "criticalityScores": [...],
    "nodeIds": [...]
  },
  "averageRisk": 0.65
}
```

#### 6. Training History
```http
GET /training/history
```

**Response**:
```json
{
  "epochs": [1, 2, 3, ..., 50],
  "total_loss": [10375, 10366, 10054, ..., 9055],
  "price_loss": [10375, 10366, 10054, ..., 9055]
}
```

---

## 🐛 Troubleshooting

### Issue 1: "Backend not found" Error

**Error Message**:
```
Error: Failed to load model information. Make sure the backend is running.
```

**Causes**:
- Backend server not started
- Backend running on wrong port
- Firewall blocking connection

**Solutions**:

```bash
# Solution 1: Start backend
python frontend/backend_server.py

# Solution 2: Check if port 8000 is in use
netstat -ano | findstr :8000

# Solution 3: Use different port
python -m uvicorn frontend.backend_server:app --port 8001

# Solution 4: Disable firewall temporarily (Windows)
netsh advfirewall set allprofiles state off
```

### Issue 2: Module Not Found Error

**Error**:
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution**:
```bash
pip install fastapi uvicorn torch

# Or install from requirements
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; print(fastapi.__version__)"
```

### Issue 3: Port Already in Use

**Error** (Port 5173):
```
error when starting dev server:
Error: listen EADDRINUSE: address already in use :::5173
```

**Solution**:
```bash
# Windows: Find and kill process
netstat -ano | findstr :5173
taskkill /PID <PID> /F

# Or use different port
npm run dev -- --port 5174
```

### Issue 4: Slow File Upload

**Problem**: File upload taking too long

**Solutions**:
1. Use smaller files (< 5MB)
2. Reduce number of features
3. Check internet connection
4. Check backend logs for errors

```bash
# Check backend logs
tail -f server.log
```

### Issue 5: CORS Errors

**Error**:
```
Access to XMLHttpRequest at 'http://localhost:8000/...'
from origin 'http://localhost:5173' has been blocked by CORS policy
```

**Solution**:
Backend already has CORS enabled. If error persists:

1. Verify backend is running
2. Check frontend proxy in vite.config.ts
3. Restart both servers

```bash
# Restart everything
# Terminal 1
npm run dev -- --port 5173

# Terminal 2
python frontend/backend_server.py
```

### Issue 6: CSV File Validation Error

**Error**: "File size must be less than 10MB" or feature mismatch

**Solutions**:
1. Ensure file is valid CSV
2. Check file size < 10MB
3. Verify numeric values only
4. No empty rows

**Valid CSV**:
```csv
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180
```

### Issue 7: TypeScript Compilation Errors

**Error**:
```
TS2339: Property 'x' does not exist on type 'y'
```

**Solution**:
```bash
# Type check
npm run type-check

# Fix errors shown, then rebuild
npm run build
```

### Issue 8: Vite Build Failures

**Error**: Build fails during `npm run build`

**Solutions**:
```bash
# Clear cache
rm -r .vite

# Reinstall
npm install

# Try build again
npm run build
```

---

## 🚀 Production Deployment

### Step 1: Build Frontend

```bash
cd frontend
npm run build
```

Output: Creates `dist/` folder with optimized bundle

### Step 2: Deploy Frontend

**Option A: Vercel (Easiest)**
```bash
npm install -g vercel
cd frontend
vercel
```

**Option B: Netlify**
1. Connect GitHub repo
2. Build command: `npm run build`
3. Publish directory: `dist`

**Option C: Self-hosted**
```bash
# Copy dist folder to web server
scp -r dist/ user@server:/var/www/html/

# Or use nginx
docker run -p 80:80 -v $(pwd)/dist:/usr/share/nginx/html nginx
```

### Step 3: Deploy Backend

**Option A: Heroku**
```bash
heroku create ht-hgnn-api
git push heroku main
```

**Option B: AWS**
```bash
# Create EC2 instance
# Install Python and dependencies
pip install -r requirements.txt

# Run with gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 frontend.backend_server:app
```

**Option C: Docker**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "frontend.backend_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 4: Environment Setup

Create `.env` file:
```bash
VITE_API_URL=https://your-api.com
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

### Step 5: Database & Model Management

Ensure model file is accessible:
```bash
# Production model path
/var/models/ht_hgnn_model.pt

# Update backend_server.py
model_path = Path("/var/models/ht_hgnn_model.pt")
```

---

## 📝 Development Workflow

### Adding a New Component

1. Create component in `src/components/`
2. Export in type definitions
3. Import in App.tsx
4. Add to routing/navigation

### Adding a New API Endpoint

1. Add endpoint to `backend_server.py`
2. Create service method in `inferenceService.ts`
3. Add TypeScript types
4. Use in components

### Code Style

```typescript
// Components
export const ComponentName: React.FC<Props> = ({ prop1, prop2 }) => {
  return <div>...</div>;
};

// Services
async function apiCall(): Promise<ReturnType> {
  // implementation
}

// Styling
className="flex items-center justify-between px-4 py-2 rounded-lg bg-blue-50 hover:bg-blue-100"
```

---

## ✅ Verification Checklist

Before submitting, verify:

- [ ] Frontend runs: `npm run dev` ✓
- [ ] Backend runs: `python frontend/backend_server.py` ✓
- [ ] Can access http://localhost:5173 ✓
- [ ] Model info loads from API ✓
- [ ] Can upload CSV file ✓
- [ ] Results display correctly ✓
- [ ] All pages accessible ✓
- [ ] Responsive on mobile ✓
- [ ] No console errors (F12) ✓
- [ ] No network errors (Network tab) ✓

---

## 📚 Resources

- [React Docs](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Vite Guide](https://vitejs.dev)
- [FastAPI](https://fastapi.tiangolo.com)
- [Recharts](https://recharts.org)

---

## 🎉 You're Ready!

The frontend is fully functional and ready for:
1. Development & testing
2. Academic submission
3. Production deployment
4. Integration with live data

**Happy coding!** 🚀
