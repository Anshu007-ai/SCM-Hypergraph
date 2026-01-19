# Frontend Setup & Installation Guide

## Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
cd frontend
npm install
```

This will install:
- React 18 + TypeScript
- Vite (bundler)
- Tailwind CSS (styling)
- Recharts (charts)
- Lucide React (icons)
- Axios (HTTP client)

### Step 2: Start Backend Server

```bash
# Open a new terminal
cd d:\College\Final year Project\SChypergraph

# Install Python dependencies (if not already installed)
pip install fastapi uvicorn torch

# Start the backend
python frontend/backend_server.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Start Frontend Development Server

```bash
# In the first terminal
cd frontend
npm run dev
```

Expected output:
```
VITE v4.5.0  ready in 234 ms

➜  Local:   http://localhost:5173/
```

### Step 4: Open in Browser

Visit: **http://localhost:5173**

---

## 📋 Installation Checklist

- [ ] Node.js 18+ installed (`node --version`)
- [ ] Python 3.8+ installed (`python --version`)
- [ ] Dependencies installed (`npm install`)
- [ ] Backend dependencies installed (`pip install fastapi uvicorn`)
- [ ] Backend running on port 8000
- [ ] Frontend running on port 5173
- [ ] Browser opens http://localhost:5173

---

## 🔧 Available Commands

### Frontend Commands

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type check
npm run type-check

# Lint
npm run lint
```

### Backend Commands

```bash
# Run development server with auto-reload
python -m uvicorn frontend.backend_server:app --reload

# Run production server
python -m uvicorn frontend.backend_server:app --host 0.0.0.0 --port 8000

# Run with custom settings
python -m uvicorn frontend.backend_server:app --reload --host 0.0.0.0 --port 8000 --log-level info
```

---

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.tsx              # Navigation bar
│   │   ├── FileUpload.tsx          # CSV uploader with drag-drop
│   │   ├── RiskDashboard.tsx       # Risk charts & metrics
│   │   └── ModelStats.tsx          # Training performance
│   ├── services/
│   │   └── inferenceService.ts    # API client
│   ├── types/
│   │   └── index.ts               # TypeScript interfaces
│   ├── App.tsx                     # Main component
│   ├── main.tsx                    # Entry point
│   └── index.css                   # Styles
├── index.html
├── vite.config.ts
├── tsconfig.json
├── package.json
├── backend_server.py               # FastAPI server
└── README.md
```

---

## 🌐 Pages Overview

### 1. Dashboard (Home)
- **Path**: `/` or `home`
- **Features**:
  - Model info card
  - CSV file uploader (drag & drop)
  - Training metrics
- **Action**: Upload CSV to analyze

### 2. Analysis
- **Path**: `analysis`
- **Features**:
  - Risk distribution charts
  - Critical nodes ranking
  - Risk summary cards
  - Recommendations
- **Requires**: File upload first

### 3. Model
- **Path**: `model`
- **Features**:
  - Loss convergence chart
  - Training metrics
  - Performance statistics
  - Training time/memory

### 4. About
- **Path**: `about`
- **Features**:
  - Architecture explanation
  - Key features list
  - Use cases
  - Component descriptions

---

## 🎯 Using the Frontend

### Analyzing Data

1. **Go to Dashboard** (default page)
2. **Upload CSV file**:
   - Drag & drop file onto upload area, OR
   - Click to browse and select
   - File must be CSV format
   - Max size: 10MB

3. **Wait for processing**:
   - Backend processes file
   - Shows "Processing..." indicator
   - Redirects to Analysis page

4. **Review Results**:
   - Check risk metrics
   - Identify critical nodes
   - Read recommendations

### Interpreting Results

**Risk Levels**:
- 🟢 **LOW**: < 25% criticality
- 🟡 **MEDIUM**: 25-50% criticality
- 🟠 **HIGH**: 50-75% criticality
- 🔴 **CRITICAL**: > 75% criticality

**Metrics**:
- **Total Nodes**: Number of supply chain entities analyzed
- **Average Risk**: Overall supply chain risk score (%)
- **Critical Nodes**: Count of high-risk suppliers
- **High Risk**: Count of medium-high risk suppliers

---

## 📊 Example CSV Format

For predictions, use CSV with numerical features:

```csv
feature_1,feature_2,feature_3,...,feature_18
100,200,300,...,400
150,250,350,...,450
...
```

Headers are optional. The system expects numeric values.

---

## 🔧 Configuration

### Backend API URL

Edit `src/services/inferenceService.ts` line 12:

```typescript
constructor(baseURL: string = 'http://localhost:8000')
```

### API Proxy

Edit `vite.config.ts` for proxy settings:

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
}
```

---

## 🐛 Common Issues & Solutions

### Issue: "Backend not found"
```
Error: Failed to load model information. Make sure the backend is running.
```

**Solution**:
```bash
# Terminal 2
cd d:\College\Final year Project\SChypergraph
python frontend/backend_server.py
```

### Issue: Port already in use

**Solution**:
```bash
# Find process using port 5173
netstat -ano | findstr :5173

# Kill the process
taskkill /PID <PID> /F

# Or use different port
npm run dev -- --port 5174
```

### Issue: Module not found

**Solution**:
```bash
rm -r node_modules
npm install
```

### Issue: TypeScript errors

**Solution**:
```bash
npm run type-check
# Fix errors shown
```

### Issue: Slow file upload

**Solution**:
1. Use smaller CSV files (< 5MB)
2. Check internet connection
3. Reduce number of features (max 18)

---

## 🚀 Production Build

```bash
# Build optimized bundle
npm run build

# Output in dist/ folder
# Upload dist/ to hosting service
```

### Hosting Options

1. **Vercel** (Recommended for React)
   ```bash
   npm install -g vercel
   vercel
   ```

2. **Netlify**
   - Connect GitHub repo
   - Set build command: `npm run build`
   - Set publish directory: `dist`

3. **GitHub Pages**
   ```bash
   npm run build
   # Push dist/ to gh-pages branch
   ```

---

## 📱 Responsive Design

The frontend is fully responsive:
- **Desktop**: Full-width layout with 2-column grids
- **Tablet**: Adaptive layout, 1-2 columns
- **Mobile**: Single column, optimized touch targets

Test responsiveness:
1. Open DevTools (F12)
2. Toggle Device Toolbar (Ctrl+Shift+M)
3. Test different screen sizes

---

## 🔒 Security Notes

### Frontend
- ✅ Input validation on file upload
- ✅ CORS headers validation
- ✅ Type-safe with TypeScript
- ✅ Secure API calls with HTTPS (in production)

### Backend
- ✅ CORS middleware enabled
- ✅ Input validation on all endpoints
- ✅ Error handling
- ✅ Logging for debugging

### Before Production
1. [ ] Enable HTTPS/TLS
2. [ ] Add authentication
3. [ ] Implement rate limiting
4. [ ] Add request validation
5. [ ] Set up monitoring
6. [ ] Use environment variables
7. [ ] Add request signing

---

## 📚 API Integration Details

### InferenceService

Located in `src/services/inferenceService.ts`

Main methods:
```typescript
// Get model information
getModelInfo(): Promise<ModelInfo>

// Make predictions
predict(input: PredictionInput): Promise<PredictionOutput>

// Analyze supply chain
analyze(input: PredictionInput): Promise<AnalysisResult>

// Upload file and predict
uploadAndPredict(file: File): Promise<PredictionOutput>

// Health check
getHealth(): Promise<{ status: string }>

// Get training history
getTrainingHistory(): Promise<Record<string, number[]>>
```

### Usage Example

```typescript
import inferenceService from './services/inferenceService';

// In component
const predictions = await inferenceService.predict({
  nodeFeatures: [[1, 2, 3, ...], ...],
  incidenceMatrix: [[0, 1, ...], ...]
});
```

---

## 📞 Support

### Check These First
1. Is backend running on port 8000?
2. Is frontend running on port 5173?
3. Are dependencies installed (`npm install`)?
4. Check browser console (F12) for errors
5. Check terminal output for warnings

### Debug Mode
```bash
# Enable verbose logging
npm run dev -- --debug

# Check API calls in DevTools Network tab
```

---

## ✅ Next Steps

1. **Install & Run**
   ```bash
   npm install && npm run dev
   ```

2. **Start Backend**
   ```bash
   python frontend/backend_server.py
   ```

3. **Open http://localhost:5173**

4. **Upload test CSV file**

5. **Review results**

---

**You're all set!** 🎉

The frontend is ready for supply chain risk analysis. For questions, refer to the README.md or check the API documentation.
