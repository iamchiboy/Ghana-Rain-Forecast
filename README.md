# Ghana Rain Nowcasting (30-Minute Prediction)

This project is a real-time machine learning system that predicts whether it will rain 30 minutes ahead in Ghana using live weather data. The goal is to improve short-term climate awareness and demonstrate the full ML pipeline from data collection to deployment.

## Overview

This project collects real-time weather data, processes it, trains a predictive model, and provides rainfall forecasting through an interactive dashboard.

## Features

- **Real-time Data Collection**: Fetches weather data from OpenWeatherMap API
- **Data Preprocessing**: Cleans and prepares weather data for model training
- **Machine Learning Model**: Trains a predictive model to forecast rainfall
- **Interactive Dashboard**: Streamlit-based dashboard for viewing predictions
- **Automated Pipeline**: Scripts for the complete ML workflow

## System Architecture

OpenWeather API
      ↓
Data Collector (every 10 minutes)
      ↓
Feature Engineering & Labeling
      ↓
ML Model (Random Forest)
      ↓
Prediction (Rain / No Rain)
      ↓
Streamlit Dashboard

## Project Structure

```
Ghana Rain Nowcast/
├── app/
│   └── dashboard.py          # Streamlit dashboard application
├── src/
│   ├── collect_data.py       # Fetch weather data from API
│   ├── preprocess.py         # Data cleaning and preprocessing
│   ├── train_model.py        # Model training script
│   └── predict.py            # Make predictions on new data
├── data/
│   ├── raw/                  # Raw weather data from API
│   └── processed/            # Processed data for modeling
├── requirements.txt          # Python dependencies
├── openweather.env          # API configuration (not in git)
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Ghana Rain Forecast"
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or: source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API credentials:
   - Create an `openweather.env` file in the root directory
   - Add your OpenWeatherMap API key:
   ```
   OPENWEATHER_API_KEY=your_api_key_here
   ```
   - Get a free API key at [openweathermap.org](https://openweathermap.org/api)

## Usage

### Collect Weather Data
```bash
python src/collect_data.py
```
Fetches current weather data for Accra and saves to `data/raw/weather_accra.csv`

### Preprocess Data
```bash
python src/preprocess.py
```
Cleans and transforms raw data, saves processed data to `data/processed/`

### Train Model
```bash
python src/train_model.py
```
Trains the rainfall prediction model

### Make Predictions
```bash
python src/predict.py
```
Uses the trained model to predict rainfall

### View Dashboard
```bash
streamlit run app/dashboard.py
```
Opens an interactive web dashboard to visualize predictions and weather data

## Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning model
- **requests**: API calls to OpenWeatherMap
- **streamlit**: Interactive dashboard
- **python-dotenv**: Environment variable management

## Requirements

- Python 3.7+
- OpenWeatherMap API key (free tier available)
- Internet connection for API calls

## Future Improvements

- [ ] Add multi-city forecasting
- [ ] Implement LSTM/neural network models
- [ ] Deploy dashboard to cloud (Heroku, AWS, etc.)
- [ ] Add seasonal analysis
- [ ] Implement automated data collection scheduling

## License

MIT

## Author

Lucky Johnson Okoro

## Contact

[luckyjohnson65@yahoo.com]
