# Ghana Rain Nowcasting (30-Minute Prediction)

**An AI-powered machine learning system for real-time rainfall forecasting in Accra, Ghana**

This project collects real-time weather data, engineers predictive features, trains a machine learning model, and provides 30-minute rain forecasts through an interactive web dashboard.

## 🌟 Features

- ✅ **Real-time Data Collection** - Automatic weather data fetching every 10 minutes
- ✅ **Advanced Feature Engineering** - Rolling windows, lag features, derived meteorological indicators
- ✅ **Robust ML Model** - Random Forest with cross-validation and performance metrics
- ✅ **Prediction Confidence** - Probability scores and confidence levels for each forecast
- ✅ **Interactive Dashboard** - Beautiful Streamlit interface with charts and metrics
- ✅ **Production-Ready** - Error handling, logging, validation, and monitoring
- ✅ **Secure** - Environment variable management for API keys

## 🏗️ System Architecture

```
OpenWeather API (10-min interval)
        ↓
Data Collection & Validation
        ↓
Feature Engineering (rolling, lag, derived features)
        ↓
Data Cleaning & Preprocessing
        ↓
Train/Test Split with Stratification
        ↓
Random Forest Classifier (300 trees)
        ↓
Cross-Validation & Metrics
        ↓
Live Prediction Module
        ↓
Streamlit Dashboard (Real-time visualization)
```

## 📁 Project Structure

```
Ghana Rain Forecast/
├── app/
│   └── dashboard.py              # Streamlit web dashboard
├── src/
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration
│   ├── logger.py                 # Logging setup
│   ├── collect_data.py           # API data collection with validation
│   ├── preprocess.py             # Feature engineering & preprocessing
│   ├── train_model.py            # Model training with validation
│   ├── predict.py                # Prediction module with confidence scores
│   ├── rain_model.pkl            # Trained model (generated)
│   ├── model_metrics.json        # Model evaluation metrics (generated)
│   └── feature_importance.png    # Feature importance chart (generated)
├── data/
│   ├── raw/
│   │   └── weather_accra.csv     # Raw API data (not in git)
│   └── processed/
│       └── weather_features.csv  # Engineered features (not in git)
├── logs/
│   └── app.log                   # Application logs (generated)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── openweather.env               # API key (not in git)
└── README.md                     # This file
```

## 📋 Installation

### Prerequisites
- Python 3.8+
- OpenWeatherMap API key (free at https://openweathermap.org/api)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "Ghana Rain Forecast"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or: source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   ```bash
   # Create openweather.env file with:
   OPENWEATHER_API_KEY=your_api_key_here
   ```

## 🚀 Quick Start

### 1. Start Data Collection (Run in background)
```bash
python -m src.collect_data
```
Collects weather data every 10 minutes. Let it run for at least 2-3 hours to gather sufficient training data.

### 2. Preprocess Data
```bash
python -m src.preprocess
```
Engineers features and prepares data for model training.

### 3. Train Model
```bash
python -m src.train_model
```
Trains Random Forest classifier with cross-validation and generates performance metrics.

### 4. View Predictions
```bash
streamlit run app/dashboard.py
```
Launches the interactive dashboard at `http://localhost:8501`

### 5. Make Single Prediction
```bash
python -m src.predict
```
Shows current 30-minute rain forecast in terminal.

## 📊 Model Performance

The model is evaluated using:
- **Accuracy**: Overall correctness of predictions
- **F1-Score**: Balance between precision and recall (important for imbalanced data)
- **Precision**: How often rain prediction is correct
- **Recall**: Ability to catch actual rain events
- **ROC-AUC**: Model discrimination ability
- **Cross-Validation**: 5-fold stratified validation

## 🔍 Feature Engineering

The model uses:

### Raw Features
- Temperature (°C)
- Humidity (%)
- Pressure (hPa)
- Wind Speed (m/s)
- Wind Direction (degrees)
- Cloud Cover (%)
- Visibility (m)
- Hourly Rainfall (mm)

### Engineered Features
- **Rolling Averages**: 3-period rolling mean and std dev
- **Lag Features**: Previous 1 and 2 timesteps
- **Derived Features**:
  - Humidity change rate
  - Pressure drop (indicator of rain)
  - Temperature change
  - Dew point approximation

## ⚙️ Configuration

Edit `src/config.py` to customize:
- API settings (timeout, retries, intervals)
- Data paths
- Feature engineering parameters
- Model hyperparameters
- Validation thresholds
- Dashboard settings

## 📈 Dashboard Features

- **Real-time Forecast**: Current 30-minute rain prediction
- **Prediction Confidence**: Probability and confidence level
- **Weather Metrics**: Current temperature, humidity, wind, pressure
- **Trend Charts**: Interactive Plotly charts showing:
  - Temperature trends
  - Humidity & Pressure
  - Rainfall history
- **Model Info**: Performance metrics and feature importance

## 🔒 Security & Best Practices

- ✅ API keys stored in `.env` file (not committed)
- ✅ Comprehensive logging for debugging
- ✅ Data validation at collection and preprocessing stages
- ✅ Error handling with retry logic
- ✅ Stratified train/test split to handle class imbalance
- ✅ Cross-validation to prevent overfitting

## 📝 Logs

Application logs are saved to `logs/app.log` with:
- Timestamp
- Module name
- Log level (INFO, DEBUG, ERROR, WARNING)
- Descriptive messages

## 🐛 Troubleshooting

### No data collected
- Verify API key in `openweather.env`
- Check internet connection
- Ensure data directory exists: `mkdir -p data/raw data/processed`

### Preprocessing fails
- Ensure at least 50 data points: `python -m src.collect_data` for ~8 hours
- Check `logs/app.log` for specific errors

### Dashboard not loading predictions
- Run preprocessing first: `python -m src.preprocess`
- Verify processed data exists: `data/processed/weather_features.csv`

### Model training fails
- Need minimum 100 data points
- Check for data quality issues in logs
- Ensure balanced dataset (both rain and no-rain samples)

## 📚 Data Requirements

- **Minimum data points**: 100 samples
- **Collection interval**: 10 minutes
- **Recommended history**: 2-3 weeks for production
- **Features**: Must have timestamp, weather metrics, and rain indicator

## 🔄 Automated Workflow

For production deployment, consider:

```bash
# Collect data continuously (background)
nohup python -m src.collect_data > logs/collect.log 2>&1 &

# Schedule preprocessing and training (daily)
# Use cron (Linux/Mac) or Task Scheduler (Windows)

# Run dashboard
streamlit run app/dashboard.py
```

## 📊 Model Improvement Ideas

1. **Add more weather stations** - Aggregate data from multiple locations
2. **Ensemble methods** - Combine multiple models
3. **Deep learning** - LSTM/GRU for sequence modeling
4. **External features** - Seasonal indicators, time of day, day of week
5. **Hyperparameter tuning** - GridSearch or Bayesian optimization
6. **Class balancing** - SMOTE or class weights

## 👨‍💻 Development

To contribute improvements:

1. Create a feature branch
2. Test changes locally
3. Update documentation
4. Submit pull request

## 📄 License

[Your License Here]

## 🤝 Support

For issues or questions, please:
- Check the logs: `cat logs/app.log`
- Review troubleshooting section above
- Open an issue on GitHub

## 📞 Contact

**Ghana Rain Nowcasting Team**
- Email: [your-email]
- GitHub: [your-github]

---

**Last Updated**: February 2026

**Status**: ✅ Production Ready
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
