# Streamlit Demo App

Interactive web demo for the Regression Language Model.

## Running the App

```bash
# From the project root directory
streamlit run app_demo/streamlit_app.py
```

Or from the `app_demo` directory:

```bash
cd app_demo
streamlit run streamlit_app.py
```

## Features

- **Real-time Predictions**: Enter text and get instant numerical predictions
- **Uncertainty Visualization**: See confidence intervals and uncertainty estimates
- **Token Importance**: Visualize which tokens the model focuses on
- **Multiple Domains**: Examples for restaurant reviews, financial news, and real estate
- **Interactive Plots**: Plotly-based visualizations

## Usage

1. Start the app with the command above
2. Enter text in the input box or click example buttons
3. Click "Predict" to get results
4. Explore the attention weights and uncertainty estimates

## Requirements

All dependencies are installed in the virtual environment. Make sure to activate it:

```bash
# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

