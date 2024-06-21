# Import necessary libraries
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg (non-interactive backend)
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg (non-interactive backend)
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf



# Set up tensorflow probability and STS
tf.enable_v2_behavior()
tfd = tfp.distributions
sts = tfp.sts

# Function to simulate ad spend data
def simulate_ad_spend_data():
    data = {
        'Date': pd.date_range(start='2020-01-01', periods=24, freq='M').tolist() * 3,
        'Channel': ['Facebook'] * 24 + ['Google'] * 24 + ['Twitter'] * 24,
        'AdSpend': np.random.randint(100, 1000, size=72)
    }
    return pd.DataFrame(data)

# Function to forecast ad spend using BSTS
def forecast_ad_spend(df, channel, forecast_months, adjust_actuals):
    monthly_data = df.groupby([pd.Grouper(key='Date', freq='M'), 'Channel']).sum().reset_index()
    monthly_data['Month_ordinal'] = monthly_data['Date'].dt.to_period('M').apply(lambda x: x.ordinal)

    channel_data = monthly_data[monthly_data['Channel'] == channel]
    observed_time_series = channel_data['AdSpend'].values.astype(np.float32)

    if adjust_actuals:
        observed_time_series[-1] *= 1.1  # Adjust the last actual by 10%

    # Define the model
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    seasonal = sts.Seasonal(num_seasons=12, observed_time_series=observed_time_series)
    model = sts.Sum([trend, seasonal], observed_time_series=observed_time_series)

    # Fit the model to the data
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

    num_variational_steps = 200
    optimizer = tf.optimizers.Adam(learning_rate=0.1)

    @tf.function(experimental_compile=True)
    def train():
        elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=model.joint_log_prob(observed_time_series=observed_time_series),
            surrogate_posterior=variational_posteriors,
            optimizer=optimizer,
            num_steps=num_variational_steps)
        return elbo_loss_curve

    elbo_loss_curve = train()

    # Sample from the variational posterior
    q_samples = variational_posteriors.sample(50)

    # Forecast future ad spend
    future_dates = pd.date_range(start=channel_data['Date'].max(), periods=forecast_months+1, freq='M')[1:]
    future_month_ordinals = np.array([date.to_period('M').ordinal for date in future_dates]).astype(np.float32)

    predictive_dist = sts.forecast(model, observed_time_series=observed_time_series, parameter_samples=q_samples, num_steps_forecast=forecast_months)

    forecast_mean = predictive_dist.mean().numpy().flatten()
    forecast_stddev = predictive_dist.stddev().numpy().flatten()

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_AdSpend': forecast_mean,
        'Predicted_AdSpend_Lower': forecast_mean - 1.96 * forecast_stddev,
        'Predicted_AdSpend_Upper': forecast_mean + 1.96 * forecast_stddev
    })

    return forecast_df, observed_time_series

# Main Streamlit app function
def main():
    st.title('Ad Spend Forecast App')
    st.subheader('Forecasting Ad Spend for Different Channels using BSTS')

    # Simulate ad spend data
    df = simulate_ad_spend_data()

    # Display sidebar for user inputs
    st.sidebar.title('User Inputs')
    channels = df['Channel'].unique()
    selected_channel = st.sidebar.selectbox("Select Channel:", channels)
    forecast_months = st.sidebar.slider("Select Forecast Months:", min_value=1, max_value=12, value=6)

    # Checkbox to adjust actuals
    adjust_actuals = st.sidebar.checkbox("Adjust Actuals")

    # Forecast and display results based on user inputs
    if st.sidebar.button("Forecast"):
        st.write(f"Forecasting for {selected_channel} channel over {forecast_months} months.")
        forecast_data, observed_actuals = forecast_ad_spend(df, selected_channel, forecast_months, adjust_actuals)

        # Plot forecasted data
        fig, ax = plt.subplots(figsize=(10, 6))
        historical_data = df[df['Channel'] == selected_channel]
        ax.plot(historical_data['Date'], historical_data['AdSpend'], label='Historical')
        ax.plot(forecast_data['Date'], forecast_data['Predicted_AdSpend'], label='Forecasted')
        ax.fill_between(forecast_data['Date'], forecast_data['Predicted_AdSpend_Lower'], forecast_data['Predicted_AdSpend_Upper'], color='gray', alpha=0.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Ad Spend')
        ax.set_title(f'{selected_channel} Ad Spend Forecast')
        ax.legend()
        st.pyplot(fig)

        # Display lower and upper bounds as table
        st.subheader("Lower and Upper Bounds")
        bounds_df = pd.DataFrame({
            'Date': forecast_data['Date'],
            'Lower Bound': forecast_data['Predicted_AdSpend_Lower'],
            'Upper Bound': forecast_data['Predicted_AdSpend_Upper']
        })
        st.write(bounds_df)

        # Display actuals and adjusted actuals
        st.subheader("Actuals vs Adjusted Actuals")
        actuals_df = pd.DataFrame({
            'Date': historical_data['Date'],
            'Actuals': observed_actuals
        })
        st.write(actuals_df)

if __name__ == '__main__':
    main()
