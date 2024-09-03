import pandas as pd


# Load the data from a CSV file
df = pd.read_csv('D:\\major project\\major project final defense\\Vehicle Data', parse_dates=['Time'])

# Display the first few rows
print(df.head())

# Set 'Time' as the DataFrame index
df.set_index('Time', inplace=True)

# Resample the data to hourly intervals and count the number of arrivals
hourly_arrivals = df.resample('H').count()

# Rename the column to 'Arrival_Count'
hourly_arrivals.rename(columns={'Vehicle_ID': 'Arrival_Count'}, inplace=True)

print(hourly_arrivals.head())


import matplotlib.pyplot as plt

# Plot the hourly arrival rate
hourly_arrivals['Arrival_Count'].plot(figsize=(10, 6), marker='o', linestyle='-')
plt.title('Hourly Vehicle Arrival Rate')
plt.xlabel('Time')
plt.ylabel('Number of Arrivals')
plt.grid(True)
plt.show()


peak_hour = hourly_arrivals['Arrival_Count'].idxmax()
peak_value = hourly_arrivals['Arrival_Count'].max()
print(f"The peak arrival hour is {peak_hour} with {peak_value} vehicles.")



from scipy.stats import poisson

# Calculate the mean arrival rate per hour
mean_arrival_rate = hourly_arrivals['Arrival_Count'].mean()

# Generate a Poisson distribution based on the mean arrival rate
poisson_dist = poisson.pmf(range(0, max(hourly_arrivals['Arrival_Count'])+1), mean_arrival_rate)

# Plot the Poisson distribution
plt.bar(range(0, len(poisson_dist)), poisson_dist, alpha=0.6, color='blue')
plt.title('Poisson Distribution of Arrival Rates')
plt.xlabel('Number of Arrivals')
plt.ylabel('Probability')
plt.show()



from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model
model = ARIMA(hourly_arrivals['Arrival_Count'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast the next 24 hours
forecast = model_fit.forecast(steps=24)

# Plot the forecast
plt.plot(hourly_arrivals.index, hourly_arrivals['Arrival_Count'], label='Historical')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title('Forecast of Vehicle Arrivals')
plt.xlabel('Time')
plt.ylabel('Number of Arrivals')
plt.legend()
plt.show()
