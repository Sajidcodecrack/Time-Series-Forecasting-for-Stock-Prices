# Time-Series-Forecasting-for-Stock-Prices

This repository contains the code and resources for a time-series forecasting project focused on predicting stock prices. The project implements and compares two distinct models: a traditional statistical ARIMA model and a deep learning Bidirectional LSTM model. The analysis is performed on historical daily stock price data for Tesla (TSLA).

The project culminates in an interactive web application deployed on Hugging Face Spaces, allowing users to generate future price forecasts.

##  Live Demo
You can try the live LSTM forecasting model here:

## Hugging Face Space: [DataSynthis_ML_JobTask](#) <!-- Replace this link with your actual Space URL -->

---

###  Table of Contents

- [Project Overview](#project-overview)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [File Structure](#file-structure)
- [Methodology](#methodology)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The primary objective of this project is to forecast future stock values using time-series analysis. The project fulfills the following requirements:

### Data Preprocessing
- Cleaning and preparing the time-series data for modeling.

### Dual Model Implementation
- A traditional **ARIMA (AutoRegressive Integrated Moving Average)** model.
- A deep learning **Bidirectional LSTM (Long Short-Term Memory)** model.

### Robust Evaluation
- Using a **rolling window evaluation** to measure forecast accuracy and model generalization over different time periods.

### Comparative Analysis
- Comparing the performance of the ARIMA and LSTM models using **RMSE** and **MAPE** metrics.

### Interactive Application
- A user-friendly web interface built with **Gradio** to interact with the trained LSTM model.

---

## Model Performance

A rolling window evaluation was performed to test the models' predictive power across different segments of the dataset. The LSTM model consistently outperformed the ARIMA model, demonstrating better generalization.

The average performance metrics across all rolling windows are summarized below:

| Model  | Average RMSE | Average MAPE (%) |
|--------|--------------|------------------|
| **LSTM** | 81.85        | 12.82%           |
| **ARIMA**| 129.39       | 19.34%           |

The **lower RMSE** and **MAPE** values indicate that the LSTM model's forecasts were significantly more accurate.

---

## Tech Stack

- **Programming Language**: Python 3.9
- **Data Analysis**: Pandas, NumPy
- **ML / DL Frameworks**: TensorFlow (Keras), Scikit-learn, Statsmodels
- **Web Framework**: Gradio
- **Development Environment**: Jupyter Notebook, Google Colab
- **Deployment**: Hugging Face Spaces

---

## Setup and Installation

To run this project locally, follow these steps:

### Clone the repository:

```bash
git clone https://github.com/YourUsername/DataSynthis_ML_JobTask.git
cd DataSynthis_ML_JobTask
```
## Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
## Install the required libraries:
```
pip install -r requirements.txt
```
## Run the Gradio application:
```
python app.py
```
## The application will launch on a local URL (e.g., http://127.0.0.1:7860
## File Structure
```
.
├── DataSynthis_ML_JobTask.ipynb  # Jupyter notebook with model development and analysis
├── app.py                        # The Gradio application script
├── lstm_model.h5                 # The saved, trained LSTM model file
├── requirements.txt              # Python dependencies for the project
├── TSLA.csv                      # The dataset used for training and forecasting
└── README.md                     # This README file
```

## Methodology
## Data Loading and Preprocessing: 
The TSLA.csv dataset is loaded, cleaned, and the Date column is set as the index. The data is then split into training and testing sets.
## ARIMA Model:
An **ARIMA** model is implemented using the statsmodels library. It is trained on a subset of the data and evaluated.
## LSTM Model:
A **Bidirectional LSTM** model is built using TensorFlow/Keras. The data is scaled using MinMaxScaler and shaped into sequences (time steps) suitable for the network. The model is trained to predict the next day's closing price.
## Rolling Window Evaluation:
To ensure a robust comparison, a rolling window approach is used. Both models are iteratively trained and tested on sliding windows of the dataset to evaluate their performance on different time periods.
## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
