# Brent Oil Dashboard

Interactive dashboard for Brent oil price change point analysis.

## Setup & Run

### 1. Install backend dependencies
pip install flask flask-cors pandas numpy

### 2. Start the Flask server
python backend/app.py

### 3. Open the dashboard
Visit: http://127.0.0.1:5000

## Features
- Historical Brent oil price chart (1987-2022)
- Bayesian change point marker (2008-08-21, Global Financial Crisis)
- Event overlays by category: Geopolitical, OPEC, Economic, Demand
- Date range filtering
- Clickable event sidebar
- Stats header: min, max, avg price and data point count
