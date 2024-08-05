import pandas as pd
import logging
from visualization import plot_series, plot_decomposition
from modeling import decompose_series, train_arima_model, forecast_arima_model

def main():
    logging.basicConfig(level=logging.INFO)

    try:
        data = pd.read_csv('AAPL.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        df = data.set_index('Date').iloc[:-2, 0:2]

        logging.info('Data loaded and processed.')

        plot_series(df)

        decomposed = decompose_series(df['AAPL'])
        plot_decomposition(decomposed)

        arima_model = train_arima_model(df['AAPL'], order=(1, 1, 1))
        logging.info('ARIMA model trained.')

        ypred, conf_int = forecast_arima_model(arima_model, steps=2)
        logging.info('Forecasting completed.')

        # Creating a new DataFrame with the prediction values.
        Date = pd.Series(['2024-01-01', '2024-02-01'])
        price_actual = pd.Series([184.40, 185.04])
        price_predicted = pd.Series(ypred.values)
        lower_int = pd.Series(conf_int['lower AAPL'].values)
        upper_int = pd.Series(conf_int['upper AAPL'].values)

        dp = pd.DataFrame({
            'Date': Date,
            'price_actual': price_actual,
            'lower_int': lower_int,
            'price_predicted': price_predicted,
            'upper_int': upper_int
        })
        dp = dp.set_index('Date')
        dp.index = pd.to_datetime(dp.index)

        dp.to_csv('forecast.csv')
        logging.info('Forecast results saved to forecast.csv.')

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
