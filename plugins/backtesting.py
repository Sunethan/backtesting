import numpy as np


class Portfolio(object):
    def __init__(self, name, money, portfolio):
        self._name = name
        self._money = money
        self._portfolio = portfolio.copy()
        self._history = {'date': [0], 'money': [money + self._portfolio['QQQ'] * 43.669998]}

    def buy(self, date, stock_name, stock_number, price):
        self._money = self._money - stock_number * price
        self._portfolio[stock_name] = self._portfolio[stock_name] + stock_number

        while self._money < 0:
            self._money = self._money + 1 * price
            self._portfolio[stock_name] = self._portfolio[stock_name] - 1

        if date not in self._history['date']:
            self._history['date'].append(date)
            self._history['money'].append(self.money + self._portfolio[stock_name] * price)
        else:
            self._history['money'][-1] += self._portfolio[stock_name] * price

    def sell(self, date, stock_name, stock_number, price):
        self._money = self._money + stock_number * price
        self._portfolio[stock_name] = self._portfolio[stock_name] - stock_number

        while self._portfolio[stock_name] < 0:
            self._money = self._money - 1 * price
            self._portfolio[stock_name] = self._portfolio[stock_name] + 1

        if date not in self._history['date']:
            self._history['date'].append(date)
            self._history['money'].append(self.money - self._portfolio[stock_name] * price)
        else:
            self._history['money'][-1] -= self._portfolio[stock_name] * price

    @property
    def money(self):
        return self._money

    @property
    def portofolio(self):
        return self._portfolio

    @property
    def history(self):
        return self._history


class BackTesting(object):
    def __init__(self, person, **kwargs):
        self._person = person
        self._ma = kwargs['MA']
        self._tqqq = kwargs['TQQQ']
        self._qqq = kwargs['QQQ']
        self._sqqq = kwargs['SQQQ']

    def str_ma_cross_buy(self, i):
        trend = self._ma['short'][i] - self._ma['long'][i]
        cross_value = \
            (self._ma['short'][i - 1] - self._ma['long'][i - 1]) * (self._ma['short'][i] - self._ma['long'][i])
        if i == 0 and trend < 0:
            return True
        else:
            if cross_value < 0 and trend < 0:
                return True
            else:
                return False

    def act_ma_cross_buy(self, i, **kwargs):
        self._person.sell(i, 'TQQQ', kwargs['sell'], self._tqqq[i])
        self._person.buy(i, 'QQQ', int(kwargs['sell'] / 2), self._qqq[i])
        self._person.buy(i, 'SQQQ', kwargs['sell'], self._sqqq[i])

    def str_ma_cross_sell(self, i):
        trend = self._ma['short'][i] - self._ma['long'][i]
        cross_value = \
            (self._ma['short'][i - 1] - self._ma['long'][i - 1]) * (self._ma['short'][i] - self._ma['long'][i])
        if i == 0 and trend > 0:
            return True
        else:
            if cross_value < 0 and trend > 0:
                return True
            else:
                return False

    def act_ma_cross_sell(self, i, **kwargs):
        self._person.buy(i, 'QQQ', 0, self._qqq[i])
        self._person.buy(i, 'TQQQ', kwargs['buy'], self._tqqq[i])
        self._person.sell(i, 'SQQQ', kwargs['buy'], self._sqqq[i])
