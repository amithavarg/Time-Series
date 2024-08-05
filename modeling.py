import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error

def decompose_series(data, model='additive'):
    # Fill or drop missing values
    data = data.dropna()
    decomposed = seasonal_decompose(data, model=model)
    return decomposed

def train_arima_model(data, order):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(data, order=order)
    arima_result = model.fit()
    return arima_result

def forecast_arima_model(model, steps):
    forecast = model.get_forecast(steps=steps)
    predicted_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    return predicted_mean, conf_int

def plot_acf_pacf(data, lags=30):
    try:
        plt.rcParams.update({'figure.figsize':(12,6), 'figure.dpi':80})
        plot_acf(data, lags=lags)
        plot_pacf(data, lags=lags)
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting ACF and PACF: {e}")
        raise

def build_arima_model(data, order=(1,1,1)):
    try:
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        logging.error(f"Error building ARIMA model: {e}")
        raise

def evaluate_model(test, predictions):
    try:
        mae = mean_absolute_error(test, predictions)
        return mae
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise
