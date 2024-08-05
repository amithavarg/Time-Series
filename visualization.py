import seaborn as sns
import matplotlib.pyplot as plt

def plot_line(data, title='Line Plot', xlabel='Date', ylabel='Price (USD)'):
    try:
        plt.figure(figsize=(10, 5))
        plot = sns.lineplot(data=data)
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting line plot: {e}")
        raise

def plot_series(df):
    plot = sns.lineplot(data=df['AAPL'])
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    plt.show()

def plot_decomposition(decomposed):
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    residual = decomposed.resid
    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(decomposed.observed, label='Original', color='black')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color='red')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residual', color='black')
    plt.legend(loc='upper left')
    plt.show()