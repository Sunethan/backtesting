# Import the client
from td.client import TDClient
from td.orders import Order, OrderLeg
from td.enums import ORDER_TYPE, ORDER_SESSION, DURATION, ORDER_STRATEGY_TYPE, ORDER_INSTRUCTIONS, ORDER_ASSET_TYPE, QUANTITY_TYPE

import pandas as pd
import numpy as np
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models.widgets import Dropdown
from bokeh.io import curdoc
from bokeh.layouts import column

from bokeh.models import BooleanFilter, CDSView, Select, Range1d, HoverTool
from bokeh.palettes import Category20
from bokeh.models.formatters import NumeralTickFormatter
import utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (7.5, 7),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.rcParams.update({'font.size': 10})

'''
# Buy 1 shares of XYZ at the Market good for the Day.
### Initalize a new Order Object.
new_order = Order()
new_order.order_type(order_type = ORDER_TYPE.MARKET)
new_order.order_session(session = ORDER_SESSION.NORMAL)
new_order.order_duration(duration = DURATION.DAY)
new_order.order_strategy_type(order_strategy_type = ORDER_STRATEGY_TYPE.SINGLE)

### Define a new OrderLeg Object.
new_order_leg = OrderLeg()
new_order_leg.order_leg_instruction(instruction = ORDER_INSTRUCTIONS.BUY)
new_order_leg.order_leg_asset(asset_type = ORDER_ASSET_TYPE.EQUITY, symbol = 'TSM')
new_order_leg.order_leg_quantity_type(quantity_type = QUANTITY_TYPE.SHARES)
new_order_leg.order_leg_quantity(quantity=1)

new_order.add_order_leg(order_leg = new_order_leg)
'''

# Create a new session, credentials path is required.
TDSession = TDClient(
    client_id='',
    redirect_uri='https://localhost:8080',
    credentials_path='/Users/...../Stock/TD/td_state.json'
)

# Login to the session
TDSession.login()
#help(TDSession.get_price_history)

principals_data = TDSession.get_accounts()
#print(principals_data[0])
for key, value in principals_data[0]['securitiesAccount'].items():
    print(key, value)

# Grab real-time quotes for 'MSFT' (Microsoft)
msft_quotes = TDSession.get_quotes(instruments=['MSFT'])

# Grab real-time quotes for 'AMZN' (Amazon) and 'SQ' (Square)
multiple_quotes = TDSession.get_quotes(instruments=['AMZN', 'SQ'])

AMZN = TDSession.get_price_history(symbol='AMZN',
                                   period='1',
                                   period_type='year',
                                   frequency='1',
                                   frequency_type='daily')

AMAZ_data = pd.DataFrame(AMZN['candles'])
AMAZ_data['date'] = pd.to_datetime(AMAZ_data['datetime'], unit='ms')
#print(AMAZ_data)

#print(AMAZ_data['open'])
#print(AMAZ_data['open'][0])
#print(AMAZ_data['open'].iloc[-1])

#print(AMAZ_data['open'].to_list()[-1])
AMA_list = AMAZ_data['open'].to_list()
SMA = utils.simple_moving_average(AMA_list, 5)
#print(SMA)


def plot_stock_price(stock):
    p = figure(plot_width=W_PLOT, plot_height=H_PLOT, tools=TOOLS,
               title="Stock price", toolbar_location='above')

    inc = stock.data['close'] > stock.data['open']
    dec = stock.data['open'] > stock.data['close']
    view_inc = CDSView(source=stock, filters=[BooleanFilter(inc)])
    view_dec = CDSView(source=stock, filters=[BooleanFilter(dec)])

    p.segment(x0='index', x1='index', y0='low', y1='high', color=RED, source=stock, view=view_inc)
    p.segment(x0='index', x1='index', y0='low', y1='high', color=GREEN, source=stock, view=view_dec)

    p.vbar(x='index', width=VBAR_WIDTH, top='open', bottom='close', fill_color=BLUE, line_color=BLUE,
           source=stock, view=view_inc, name="price")
    p.vbar(x='index', width=VBAR_WIDTH, top='open', bottom='close', fill_color=RED, line_color=RED,
           source=stock, view=view_dec, name="price")

    p.legend.location = "top_left"
    p.legend.border_line_alpha = 0
    p.legend.background_fill_alpha = 0
    p.legend.click_policy = "mute"

    # map dataframe indices to date strings and use as label overrides
    p.xaxis.major_label_overrides = {
        i + int(stock.data['index'][0]): date.strftime('%b %d') for i, date in
        enumerate(pd.to_datetime(stock.data["date"]))
    }
    p.xaxis.bounds = (stock.data['index'][0], stock.data['index'][-1])

    # Add more ticks in the plot
    p.x_range.range_padding = 0.05
    p.xaxis.ticker.desired_num_ticks = 40
    p.xaxis.major_label_orientation = 3.14 / 4

    return p


# Define constants
W_PLOT = 1500
H_PLOT = 600
TOOLS = 'pan,wheel_zoom,reset'

VBAR_WIDTH = 0.2
RED = Category20[7][6]
GREEN = Category20[5][4]

BLUE = Category20[3][0]
BLUE_LIGHT = Category20[3][1]

ORANGE = Category20[3][2]
PURPLE = Category20[9][8]
BROWN = Category20[11][10]

stock = ColumnDataSource(data=AMAZ_data)

# update_plot()
p_stock = plot_stock_price(stock)
show(p_stock)



'''
plt.figure(1)
plt.plot(AMAZ_data['datetime'].to_list(), AMAZ_data['open'].to_list(), '-bo')
plt.plot(AMAZ_data['datetime'].to_list(), SMA, '-ro')
plt.show()
'''
