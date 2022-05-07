import pandas as pd
import numpy as np
from pandas.api.indexers import BaseIndexer


class MyIndexer(BaseIndexer):
    def get_window_bounds(
            self,
            num_values=0, min_periods=None, center=None, closed=None
    ):
        start = np.arange(num_values, dtype="int64")
        end = start + self.window_size
        if self.window_size:
            end[-self.window_size:] = num_values
        start = np.zeros(num_values, dtype="int64")
        return start, end


def myRank(df):
    ranked = df.rank(pct=True)
    return ranked.iloc[-1]


### Initial Operations
def returns(df):
    """
    close-to-close returns
    """
    return df.close / df.close.shift(1) - 1


def vwap(df):
    """
    volume-weighted average price
    """
    return (df.volume * df.close) / df.volume


def adv(df, d):
    """
    adv{d} = average daily dollar volume for the past d days
    """
    return df.volume.rolling(d).mean()


###
def rank(df):
    """
    Cross-sectional percentile rank.

    :param df:
    :return:
    """
    indexer = MyIndexer(window_size=1)
    return df.rolling(window=indexer).apply(lambda x: myRank(x))


def stddev(df, d):
    """
    Rolling standard deviation over the last d days.

    :param df:
    :param d:
    :return:
    """
    return df.rolling(d).std()


def scale(df, a=1):
    """
    rescaled x such that sum(abs(x)) = a (the default is a = 1)
    :param df:
    :param a:
    :return:
    """
    indexer = MyIndexer(window_size=1)
    return df.abs() / df.replace([np.inf, -np.inf], np.nan).abs().rolling(window=indexer).sum(skipna=True)


def product(df, d):
    """
    time-series product over the past d days
    :param df: data frame containing prices
    :param d: number of days to look back (rolling window)
    :return: Pandas series
    """
    return df.rolling(d).apply(np.product)


def decay_linear(df, d):
    """
    weighted moving average over the past d days with linearly decaying
    weights d, d – 1, …, 1 (rescaled to sum up to 1)
    :param df: data frame containing prices
    :param d: number of days to look back (rolling window)
    :return: Pandas series
    """
    return df.ewm(d).mean()


def delta(df, d):
    """
    today’s value of x minus the value of x d days ago
    """
    return df - df.shift(d)


def corr(x, y, d):
    """
    time-serial correlation of x and y for the past d days
    """
    return x.rolling(d).corr(y)


def cov(x, y, d):
    """
    time-serial covariance of x and y for the past d days
    """
    return x.rolling(d).cov(y)


def delay(df, d):
    """
    value of x d days ago
    """
    return df.shift(d)


### Time-Series Operations
def ts_max(df, d=10):
    """
    The rolling max over the last d days.

    :param df: data frame containing prices
    :param d: number of days to look back (rolling window)
    :return: Pandas series
    """
    return df.rolling(d).max()


def ts_min(df, d=10):
    """
    The rolling min over the last d days.

    :param df: data frame containing prices
    :param d: number of days to look back (rolling window)
    :return: Pandas series
    """
    return df.rolling(d).min()


def ts_argmax(df, d):
    """
    Gets the day, ts_max(x, d) occured on.

    :param df: dataframe
    :param d: number of days to look back (rolling window)
    :return: Pandas Series
    """
    return df.rolling(d).apply(np.argmax).add(1)


def ts_argmin(df, d):
    """
    Gets the day, ts_min(x, d) occured on.

    :param df: dataframe
    :param d: number of days to look back (rolling window)
    :return: Pandas Series
    """
    return df.rolling(d).apply(np.argmin).add(1)


def ts_rank(df, d):
    """
    time-series rank in the past d days

    :param df: dataframe
    :param d: number of days to look back (rolling window)
    :return: Pandas Series
    """
    return df.rolling(d).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])


def ts_sum(df, d):
    """
    time-series sum over the past d days
    """
    return df.rolling(d).sum()
