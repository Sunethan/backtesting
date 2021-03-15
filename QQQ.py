import DataAnalysis as Data
import utils
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 11,
          'legend.title_fontsize': 13,
          'figure.figsize': (13, 7),
          'axes.labelsize': 13,
          'axes.titlesize': 13,
          'xtick.labelsize': 13,
          'ytick.labelsize': 13}
pylab.rcParams.update(params)
plt.rcParams.update({'font.size': 11})

pwd = r'C:\Users\ethankao\OneDrive - Synopsys, Inc\Documents\PycharmProjects\Synopsys\Stock\data'
path = {'QQQ': pwd + r'\QQQ.csv',
        'SQQQ': pwd + r'\SQQQ.csv',
        'TQQQ': pwd + r'\TQQQ.csv'}


# Stock['QQQ'].x = Date
# Stock['QQQ'].y = Price
Stock = {ETF: Data.Read(path[ETF], xlabel='Date', ylabel='Close', floatKey='No', yType='float') for ETF in path.keys()}
Date = {ETF: [i for i in range(len(Stock[ETF].x))] for ETF in path.keys()}


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


Cash = 100000  # [USD]
Initial_stock = 10000  # [USD]
MA1 = 3  # Moving average 1
MA2 = 10  # Moving average 2
Buy_Number = 50  # [share]
Sell_Number = 10  # [share]

InitialState = {'QQQ': math.floor(Initial_stock/Stock['QQQ'].y[0]), 'SQQQ': 0, 'TQQQ': 0}
Ethan = Portfolio('Ethan', Cash, InitialState)
MA = {MA1: utils.center_smooth(Stock['QQQ'].y, MA1),
      MA2: utils.center_smooth(Stock['QQQ'].y, MA2)}

# Strategy


def cross(i):
    if i == 0:
        return True
    else:
        if (MA[MA1][i - 1] - MA[MA2][i - 1]) * (MA[MA1][i] - MA[MA2][i]) < 0:
            return True
        else:
            return False


for i, (date, price) in enumerate(zip(Date['QQQ'], Stock['QQQ'].y)):
    if cross(i):
        if MA[MA1][i] - MA[MA2][i] < 0:
            Ethan.sell(i, 'TQQQ', Sell_Number, Stock['TQQQ'].y[i])
            Ethan.buy(i, 'QQQ', int(Sell_Number / 2), Stock['QQQ'].y[i])
            Ethan.buy(i, 'SQQQ', Sell_Number, Stock['SQQQ'].y[i])
        else:
            Ethan.buy(i, 'QQQ', int(Buy_Number / 2), Stock['QQQ'].y[i])
            Ethan.buy(i, 'TQQQ', Buy_Number, Stock['TQQQ'].y[i])
            Ethan.sell(i, 'SQQQ', Buy_Number, Stock['SQQQ'].y[i])


# Plot

ColorSet = utils.color_set(path.keys())
fig, ax1 = plt.subplots()
ETF = 'QQQ'
ax1.plot(Date[ETF], Stock[ETF].y, color='b', marker='o', fillstyle='none', label=ETF)
ax1.plot(Date[ETF], utils.center_smooth(Stock[ETF].y, MA1), color='r', marker='o', fillstyle='none', linestyle='-.', label='MA(%s)' % MA1)
ax1.plot(Date[ETF], utils.center_smooth(Stock[ETF].y, MA2), color='g', marker='o', fillstyle='none', linestyle='-.', label='MA(%s)' % MA2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(Ethan.history['date'], Ethan.history['money'], color='fuchsia', marker='o', fillstyle='none', linestyle='-')
ax2.set_ylabel('Asset')
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock price')
plt.grid()
ax1.legend(loc='lower right')

ratio = Stock['QQQ'].y[-1]/Stock['QQQ'].y[0]
ax2.axhline(y=ratio * (Cash + InitialState['QQQ'] * Stock['QQQ'].y[0]), xmin=0, xmax=Date[ETF][-1], color='black', linestyle='-.')
plt.show()
