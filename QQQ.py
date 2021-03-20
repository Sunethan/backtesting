import DataAnalysis as Data
import utils
import backtesting as bt
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

Cash = 100000  # [USD]
Initial_stock = 10000  # [USD]
MA1 = 3  # Moving average 1
MA2 = 30  # Moving average 2
Buy_Number = 50  # [share]
Sell_Number = 10  # [share]

InitialState = {'QQQ': math.floor(Initial_stock/Stock['QQQ'].y[0]), 'SQQQ': 0, 'TQQQ': 0}
Ethan = bt.Portfolio('Ethan', Cash, InitialState)
MA = {'short': utils.center_smooth(Stock['QQQ'].y, MA1),
      'long': utils.center_smooth(Stock['QQQ'].y, MA2)}

# Strategy

BackTesting = bt.BackTesting(Ethan, MA=MA, QQQ=Stock['QQQ'].y, TQQQ=Stock['TQQQ'].y, SQQQ=Stock['SQQQ'].y)

for i, (date, price) in enumerate(zip(Date['QQQ'], Stock['QQQ'].y)):
    if BackTesting.str_ma_cross_buy(i):
        BackTesting.act_ma_cross_buy(i, sell=Sell_Number)
    if BackTesting.str_ma_cross_sell(i):
        BackTesting.act_ma_cross_sell(i, buy=Buy_Number)

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
