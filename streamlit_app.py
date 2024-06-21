# Import necessary libraries
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
def forecast_ad_spend(df, channel, forecast_months):
    monthly_data = df.groupby([pd.Grouper(key='Date', freq='M'), 'Channel']).sum().reset_index()
    monthly_data['Month_ordinal'] = monthly_data['Date'].dt.to_period('M').apply(lambda x: x.ordinal)

    channel_data = monthly_data[monthly_data['Channel'] == channel]
    X = channel_data[['Month_ordinal']]
    y = channel_data['AdSpend']

    model = LinearRegression()
    model.fit(X, y)

    future_dates = pd.date_range(start=channel_data['Date'].max(), periods=forecast_months, freq='M')[1:]
    future_dates_ordinal = np.array([date.to_period('M').ordinal for date in future_dates]).reshape(-1, 1)
    future_predictions = model.predict(future_dates_ordinal)

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_AdSpend': future_predictions
    })

    return forecast_df

# Main Streamlit app function
def main():
    st.title('Ad Spend Forecast App')
    st.subheader('Forecasting Ad Spend for Different Channels')

    # Simulate ad spend data
    df = simulate_ad_spend_data()

    # Display sidebar for user inputs
    st.sidebar.title('User Inputs')
    channels = df['Channel'].unique()
    selected_channel = st.sidebar.selectbox("Select Channel:", channels)
    forecast_months = st.sidebar.slider("Select Forecast Months:", min_value=1, max_value=12, value=6)

    # Forecast and display results based on user inputs
    if st.sidebar.button("Forecast"):
        st.write(f"Forecasting for {selected_channel} channel over {forecast_months} months.")
        forecast_data = forecast_ad_spend(df, selected_channel, forecast_months)

        # Plot forecasted data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df[df['Channel'] == selected_channel]['Date'], df[df['Channel'] == selected_channel]['AdSpend'], label='Historical')
        ax.plot(forecast_data['Date'], forecast_data['Predicted_AdSpend'], label='Forecasted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Ad Spend')
        ax.set_title(f'{selected_channel} Ad Spend Forecast')
        ax.legend()
        st.pyplot(fig)

if __name__ == '__main__':
    main()

