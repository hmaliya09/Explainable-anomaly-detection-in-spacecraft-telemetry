# Explainable-anomaly-detection-in-spacecraft-telemetry
Developing machine learning models for time-series anomaly detection on real NASA SMAP and MSL mission telemetry data, with explainable AI techniques to provide interpretable insights for system health monitoring.
# Spacecraft Telemetry Anomaly Detection

## Project Overview
This project develops a machine learning-based system to detect anomalies in spacecraft telemetry data from NASA’s Mars Science Laboratory (MSL) and Soil Moisture Active Passive (SMAP) missions. The goal is to identify abnormal patterns that could indicate subsystem faults or failures, ensuring mission safety and reliability.

## Key Features
- **Advanced Feature Extraction**: Combines Long Short-Term Memory (LSTM) networks, Short-Time Fourier Transform (STFT), and moving average predictors to capture diverse anomaly characteristics such as waveform, frequency, and magnitude changes.
- **Multiple Classification Models**: Implements and compares ensemble methods (AdaBoost, Random Forest), Support Vector Machines (SVM), and Neural Networks for effective binary anomaly classification.
- **Performance Metrics**: Uses precision, recall, and F0.5 score to emphasize accurate anomaly detection while minimizing false positives.
- **Explainable AI (XAI)**: Provides interpretability of model predictions to help spacecraft engineers understand and trust anomaly alerts.

## Dataset
The system is trained and validated on publicly available NASA telemetry datasets from the MSL and SMAP missions, containing labeled anomalies across multiple sensor channels.

## Usage
1. Preprocess raw telemetry data.
2. Extract features using LSTM, STFT, and moving average methods.
3. Train classification models on labeled data.
4. Evaluate model performance with cross-validation.
5. Use trained models to detect anomalies in unseen telemetry streams.

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / PyTorch (for deep learning components)
- Matplotlib (for visualization)

## Contribution
Contributions and improvements are welcome. Please open an issue or submit a pull request.
