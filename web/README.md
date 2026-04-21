<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Web Application

This contains everything you need to run the web application locally.

## Run Locally

**Prerequisites:** Node.js, Python 3.8+

### Frontend Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env.local
   ```
   Then edit `.env.local` and add your actual values:
   - `VITE_BACKEND_URL`: Backend API URL (default: `http://localhost:8000`)
   - `VITE_SEGMENT_URL`: Segmentation service URL (default: `http://localhost:8010`)

3. Run the app:
   ```bash
   npm run dev
   ```

### Backend Router (Optional)

If using the Python router:

1. Install Python dependencies:
   ```bash
   pip install fastapi httpx uvicorn
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_BACKEND_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_SEGMENT_URL` | Segmentation service URL | `http://localhost:8010` |