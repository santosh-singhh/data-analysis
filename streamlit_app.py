# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to simulate ad spend data
def simulate_ad_spend_data():
    data = {
        'Date': pd.date_range(start='2020-01-01', periods=24, freq='M').tolist() * 3,
        'Channel': ['Facebook'] * 24 + ['Google'] * 24 + ['Twitter'] * 24,
        'AdSpend': np.random.randint(100, 1000, size=72)
    }
    return pd.DataFrame(data)

# Function to forecast ad spend
def forecast_ad_spend(df):
    monthly_data = df.groupby([pd.Grouper(key='Date', freq='M'), 'Channel']).sum().reset_index()
    monthly_data['Month_ordinal'] = monthly_data['Date'].dt.to_period('M').apply(lambda x: x.ordinal)

    channels = monthly_data['Channel'].unique()
    predictions = {}

    for channel in channels:
        channel_data = monthly_data[monthly_data['Channel'] == channel]
        X = channel_data[['Month_ordinal']]
        y = channel_data['AdSpend']

        model = LinearRegression()
        model.fit(X, y)

        future_dates = pd.date_range(start=channel_data['Date'].max(), periods=7, freq='M')[1:]
        future_dates_ordinal = np.array([date.to_period('M').ordinal for date in future_dates]).reshape(-1, 1)
        future_predictions = model.predict(future_dates_ordinal)

        predictions[channel] = pd.DataFrame({
            'Date': future_dates,
            'Predicted_AdSpend': future_predictions
        })

    return predictions

# Main Streamlit app function
def main():
    st.title('Ad Spend Forecast App')
    st.subheader('Forecasting Ad Spend for Different Channels')

    # Simulate data and forecast
    df = simulate_ad_spend_data()
    predictions = forecast_ad_spend(df)

    # Display historical and forecasted data
    st.write("### Historical and Forecasted Ad Spend")

    for channel, forecast_data in predictions.items():
        st.write(f"#### {channel}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df[df['Channel'] == channel]['Date'], df[df['Channel'] == channel]['AdSpend'], label='Historical')
        ax.plot(forecast_data['Date'], forecast_data['Predicted_AdSpend'], label='Forecasted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Ad Spend')
        ax.set_title(f'{channel} Ad Spend Forecast')
        ax.legend()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
