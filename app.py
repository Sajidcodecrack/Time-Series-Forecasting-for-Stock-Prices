import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. Load the Pre-trained Model and Scaler ---

# Load the trained LSTM model from the .h5 file
model = tf.keras.models.load_model('lstm_model.h5')

# We need to re-create the scaler that was used during training.
# To do this, we load the original dataset.
data = pd.read_csv("TSLA.csv")
close_prices = data[['Close']].values

# Create and fit the scaler on the same data it was trained on.
# This ensures our predictions can be correctly inverse-transformed.
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(close_prices)

# The time step used during model training
TIME_STEP = 60

# --- 2. Define the Prediction Function ---

def predict_stock(days_to_forecast):
    """
    Takes the number of days to forecast as input, predicts future stock prices,
    and returns them in a pandas DataFrame.
    """
    
    # Get the last TIME_STEP days from the original dataset to start the prediction
    last_60_days = close_prices[-TIME_STEP:]
    
    # Scale the input data using the same scaler
    last_60_days_scaled = scaler.transform(last_60_days)
    
    # This will be our initial input for prediction
    X_input = last_60_days_scaled.reshape(1, TIME_STEP, 1)
    
    # List to store the scaled predicted prices
    predicted_prices_scaled = []
    
    # Loop to predict for the number of days specified by the user
    for i in range(int(days_to_forecast)):
        # Predict the next day's price
        predicted_price = model.predict(X_input)
        
        # Append the scaled prediction to our list
        predicted_prices_scaled.append(predicted_price[0, 0])
        
        # Update the input sequence: remove the first day and add the new prediction at the end
        new_input = np.append(X_input[0, 1:, 0], predicted_price[0, 0])
        X_input = new_input.reshape(1, TIME_STEP, 1)
        
    # Inverse transform the scaled predictions to get actual price values
    final_predictions = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))
    
    # Create a DataFrame to display the results nicely
    forecast_df = pd.DataFrame({
        "Day": range(1, int(days_to_forecast) + 1),
        "Predicted Close Price (USD)": [f"${price[0]:,.2f}" for price in final_predictions]
    })
    
    return forecast_df

# --- 3. Create the Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # Stock Price Forecaster: DataSynthis_ML_JobTask
        This application uses a trained LSTM model to forecast future stock prices for TSLA.
        Use the slider to select how many days into the future you'd like to predict.
        """
    )
    
    days_input = gr.Slider(
        minimum=1,
        maximum=30,
        step=1,
        value=7,
        label="Days to Forecast",
        info="Select the number of days you want to forecast."
    )
    
    predict_button = gr.Button("Forecast Prices")
    
    output_dataframe = gr.DataFrame(
        headers=["Day", "Predicted Close Price (USD)"],
        label="Forecasted Prices"
    )
    
    predict_button.click(
        fn=predict_stock,
        inputs=days_input,
        outputs=output_dataframe
    )

# --- 4. Launch the App ---
iface.launch()
