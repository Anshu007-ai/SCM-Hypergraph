# React + TypeScript Frontend for HT-HGNN

A modern, production-ready web application for supply chain risk analysis using the HT-HGNN (Heterogeneous Temporal Hypergraph Neural Network) model.

## 🚀 Features

- **Real-time Risk Analysis**: Upload supply chain data and get instant predictions
- **Interactive Dashboard**: Visualize risk metrics and critical nodes
- **Model Information**: View trained model details and performance metrics
- **File Upload**: CSV file support for batch predictions
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Type-Safe**: Full TypeScript support for reliability

## 📦 Tech Stack

- **Frontend**: React 18 + TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **Build Tool**: Vite
- **Backend**: FastAPI (Python)

## 🛠️ Installation

### Prerequisites

- Node.js 18+ (for frontend)
- Python 3.8+ (for backend)
- npm or yarn package manager

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The frontend will be available at `http://localhost:5173`

### Backend Setup

```bash
# Install Python dependencies
pip install fastapi uvicorn torch

# Start the backend server
python frontend/backend_server.py

# Or use uvicorn directly
uvicorn frontend.backend_server:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

## 📝 Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── Header.tsx       # Navigation header
│   │   ├── FileUpload.tsx   # CSV file uploader
│   │   ├── RiskDashboard.tsx # Risk visualization
│   │   └── ModelStats.tsx   # Model performance stats
│   ├── pages/               # Page components
│   ├── services/            # API services
│   │   └── inferenceService.ts  # Model inference API client
│   ├── types/               # TypeScript interfaces
│   │   └── index.ts         # Type definitions
│   ├── App.tsx              # Main application component
│   ├── main.tsx             # Entry point
│   └── index.css            # Global styles
├── index.html               # HTML template
├── vite.config.ts           # Vite configuration
├── tsconfig.json            # TypeScript configuration
├── package.json             # Dependencies
└── backend_server.py        # FastAPI backend server
```

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-16T15:58:00.000Z"
}
```

### Model Information
```
GET /model/info
```
Response:
```json
{
  "name": "HT-HGNN",
  "version": "1.0.0",
  "parameters": 130445,
  "device": "cuda",
  "architecture": "Heterogeneous Temporal Hypergraph NN",
  "trainingDate": "2026-01-16T15:58:00.000Z",
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

### Make Predictions
```
POST /predict
```
Request:
```json
{
  "nodeFeatures": [[18 float values], ...],
  "incidenceMatrix": [[0, 1, ...], ...]
}
```

Response:
```json
{
  "pricePredictions": [500.5, ...],
  "changePredictions": [0.05, ...],
  "criticalityScores": [0.75, ...],
  "nodeIds": ["node_0", ...]
}
```

### Analyze Supply Chain
```
POST /analyze
```
Request: Same as `/predict`

Response:
```json
{
  "timestamp": "2026-01-16T15:58:00.000Z",
  "predictions": { ... },
  "averageRisk": 0.65
}
```

### Upload File & Predict
```
POST /upload/predict
```
Request: Form data with CSV file

Response: Same as `/predict`

### Training History
```
GET /training/history
```
Response:
```json
{
  "epochs": [1, 2, ..., 50],
  "total_loss": [10375, ...],
  "price_loss": [10375, ...]
}
```

## 📊 Component Documentation

### FileUpload Component

Handles CSV file uploads with drag-and-drop support.

```tsx
<FileUpload 
  onUpload={handleFileUpload}
  isLoading={isLoading}
/>
```

Props:
- `onUpload: (file: File) => Promise<PredictionOutput>` - Callback for file upload
- `isLoading?: boolean` - Loading state

### RiskDashboard Component

Displays risk analysis results with charts and metrics.

```tsx
<RiskDashboard 
  riskAssessments={assessments}
  topCriticalNodes={topNodes}
  averageRisk={0.65}
/>
```

Props:
- `riskAssessments: RiskAssessment[]` - Array of risk assessments
- `topCriticalNodes: RiskAssessment[]` - Top critical nodes
- `averageRisk: number` - Average risk score (0-1)

### ModelStats Component

Shows model performance metrics and training statistics.

```tsx
<ModelStats 
  metrics={metrics}
  device="cuda"
/>
```

Props:
- `metrics: TrainingMetrics` - Training metrics
- `device: string` - Device used (cuda/cpu)

### Header Component

Navigation header with page routing.

```tsx
<Header 
  currentPage="home"
  onNavigate={setCurrentPage}
/>
```

Props:
- `currentPage: PageType` - Current page (home|analysis|model|about)
- `onNavigate: (page: PageType) => void` - Navigation callback

## 🎨 Styling

The application uses Tailwind CSS for styling. Customize by editing:
- `src/index.css` - Global styles
- Component files - Inline Tailwind classes

Example component styling:
```tsx
<div className="bg-white border border-gray-200 rounded-lg p-6">
  <h2 className="text-xl font-bold text-gray-900 mb-4">Title</h2>
</div>
```

## 🔐 Security Considerations

- Input validation on file uploads
- CORS middleware for API security
- Type safety with TypeScript
- Secure API endpoints

For production deployment:
1. Enable HTTPS/TLS
2. Implement authentication/authorization
3. Add rate limiting
4. Validate all inputs server-side
5. Use environment variables for configuration

## 🚀 Deployment

### Frontend Deployment (Vercel, Netlify, etc.)

```bash
npm run build
# Deploy the dist/ folder
```

### Backend Deployment (AWS, GCP, Azure, etc.)

```bash
# Create Docker container
docker build -t ht-hgnn-backend .

# Deploy to your cloud provider
docker run -p 8000:8000 ht-hgnn-backend
```

## 📖 Usage Guide

### 1. Dashboard Page
- View model information
- Upload CSV file for analysis
- Check model performance metrics

### 2. Analysis Page
- View risk predictions
- Analyze critical nodes
- Check risk distribution
- Review recommendations

### 3. Model Page
- View training metrics
- Check loss convergence
- See model specifications

### 4. About Page
- Learn about HT-HGNN architecture
- Understand use cases
- Review key features

## 🐛 Troubleshooting

### Backend Not Found
```
Error: Failed to load model information. Make sure the backend is running.
```
Solution: Start the backend server:
```bash
python frontend/backend_server.py
```

### Module Not Found
```bash
# Install missing dependencies
npm install
pip install -r requirements.txt
```

### CORS Issues
The backend server includes CORS middleware. If issues persist:
1. Check backend is running on port 8000
2. Verify frontend proxy configuration in vite.config.ts
3. Check browser console for specific error

### Slow Performance
1. Check if using GPU (CUDA) for inference
2. Reduce batch size in requests
3. Clear browser cache (Ctrl+Shift+Delete)
4. Check network tab for slow API calls

## 📚 Additional Resources

- [React Documentation](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Vite Guide](https://vitejs.dev/guide/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

## 📄 License

This project is part of a university final year project.

## 👥 Support

For issues or questions:
1. Check the troubleshooting section
2. Review API endpoint documentation
3. Check browser console for errors
4. Verify backend server is running

---

**Version**: 1.0.0  
**Last Updated**: January 16, 2026  
**Status**: Production Ready

## ✅ Visualization & UI additions

This branch adds an improved interactive visualization using `react-force-graph-2d` and animations via `animejs`, plus Radix UI primitives to support shadcn-like styling.

Install the new frontend deps after pulling changes:

```bash
cd frontend
npm install
```

If you want to extend the UI using the shadcn pattern, install the related tooling and follow the shadcn docs to scaffold components.
