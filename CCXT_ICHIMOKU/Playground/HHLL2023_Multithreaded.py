# 2 Consecutive higher highs or Lower Lows Scanner for Traders 1.0 (Inelida HigherHighs-LowerLows Scanner for Traders)
# Example of use : python HHLL2023_Multithreaded.py -e bybit -f *usdt -tf 1m,3m,5m,15m,30m,1h,2h,4h,6h,12h,1d,1w -l
# And will scan in loop and for uptrend and downtrend (if 2 higher highs or 2 lower lows + some Ichimoku filtering)
# STRICTLY EXPERIMENTAL (I'm searching for the best setup).

import sys

import os
import ccxt
import pandas as pd
from datetime import datetime
import time
import threading
import ta
import argparse
import signal
#import beepy as beep
import tweepy

from datetime import date

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', True)

enable_tweet = False


def tweet(str_to_tweet):
    if enable_tweet is False:
        return

    twitter_auth_keys = {
        "consumer_key": "",
        "consumer_secret": "",
        "access_token": "-",
        "access_token_secret": ""
    }

    auth = tweepy.OAuthHandler(
        twitter_auth_keys['consumer_key'],
        twitter_auth_keys['consumer_secret']
    )
    auth.set_access_token(
        twitter_auth_keys['access_token'],
        twitter_auth_keys['access_token_secret']
    )
    api = tweepy.API(auth)

    tweet = str_to_tweet
    status = api.update_status(status=tweet)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    os.kill(os.getpid(), 9)
    sys.exit(-888)


signal.signal(signal.SIGINT, signal_handler)


def log_to_errors(str_to_log):
    fr = open("errors.txt", "a")
    fr.write(str_to_log + "\n")
    fr.close()


def log_to_results(str_to_log):
    fr = open("results.txt", "a")
    fr.write(str_to_log + "\n")
    fr.close()


def log_to_results_evol(str_to_log):
    fr = open("results_evol.txt", "a")
    fr.write(str_to_log + "\n")
    fr.close()


def delete_results_log():
    if os.path.exists("results.txt"):
        os.remove("results.txt")


def delete_results_evol_log():
    if os.path.exists("results_evol.txt"):
        os.remove("results_evol.txt")


def delete_errors_log():
    if os.path.exists("errors.txt"):
        os.remove("errors.txt")


def log_to_results_temp(str_to_log, exchange_id):
    fr = open("results_temp_" + exchange_id + ".txt", "a")
    fr.write(str_to_log + "\n")
    fr.close()


def delete_results_temp_log(exchange_id):
    if os.path.exists("results_temp_" + exchange_id + ".txt"):
        os.remove("results_temp_" + exchange_id + ".txt")


delete_errors_log()

exchanges = {}  # a placeholder for your instances
for id in ccxt.exchanges:
    exchange = getattr(ccxt, id)
    exchanges[id] = exchange()
    # print(exchanges[id])
    try:
        ex = exchanges[id]
        # markets = ex.fetch_markets()
        # print(markets)
    except:
        continue


# function not used for now (but might be useful for a counter such as 1/xxx 2/xxx etc...)
def get_number_of_active_assets_for_exchange(exchange_id):
    nb_active_assets = 0
    arg_exchange = exchange_id
    if arg_exchange in exchanges:
        exchange = exchanges[arg_exchange]
        try:
            markets = exchange.fetch_markets()
            for oneline in markets:
                symbol = oneline['id']
                active = oneline['active']
                if active is True:
                    # print(symbol, end=' ')
                    nb_active_assets += 1
            # print("")
            # print("number of active assets =", nb_active_assets)
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.DDoSProtection):
            print("Exchange seems not available (maybe too many requests). Will stop now.")
            # exit(-10002)
            os.kill(os.getpid(), 9)
            sys.exit(-999)
            # time.sleep(5)
        except:
            print(sys.exc_info())
            exit(-10003)
    return nb_active_assets


# print(get_number_of_active_assets_for_exchange("binance"))
# exit(-1000)

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exchange", help="set exchange", required=False)
parser.add_argument('-g', '--get-exchanges', action='store_true', help="get list of available exchanges")
parser.add_argument('-a', '--get-assets', action='store_true', help="get list of available assets")
parser.add_argument('-f', '--filter-assets', help="assets filter")
parser.add_argument('-r', '--retry', action='store_true', help="retry until exchange is available (again)")
parser.add_argument('-l', '--loop', action='store_true',
                    help="scan in loop (useful for continually scan one asset or a few ones)")
parser.add_argument('-tf', '--timeframes',
                    help="the timeframe to scan for Ichimoku validations (separated by ',' if more than one)")
parser.add_argument('-up', '--up', action='store_true', help="scan for uptrend validations (default is up and down)")
parser.add_argument('-down', '--down', action='store_true',
                    help="scan for downtrend validations (default is up and down)")

args = parser.parse_args()
print("args.exchange =", args.exchange)
print("args.get-exchanges", args.get_exchanges)
print("args.get-assets", args.get_assets)
print("args.filter", args.filter_assets)
print("args.retry", args.retry)
print("args.loop", args.loop)
print("args.timeframes", args.timeframes)
print("args.up", args.up)
print("args.down", args.down)

print("INELIDA Ichimoku Scanner for Traders v1.0 - https://twitter.com/IchimokuTrader")
print("Scan started at :", str(datetime.now()))

# if a debugger is attached then set arbitrary arguments for debugging (exchange...)
if sys.gettrace() is not None:
    args.exchange = "bybit"
    args.filter_assets = "*usdt"  # "BTCPERP"
    args.loop = True
    args.timeframes = "1m,3m,5m,15m,1h,2h,4h,6h,12h,1d,1w"
    args.up = False
    args.down = False

if args.get_exchanges is True:
    for id in ccxt.exchanges:
        print(id, end=' ')
    print("")
    exit(-505)

if args.get_assets is True:
    if args.exchange is None:
        print("Please specify an exchange name")
    else:
        arg_exchange = args.exchange.lower().strip()
        if arg_exchange in exchanges:
            exchange = exchanges[arg_exchange]
            try:
                markets = exchange.fetch_markets()
                nb_active_assets = 0
                for oneline in markets:
                    symbol = oneline['id']
                    active = oneline['active']
                    if active is True:
                        print(symbol, end=' ')
                        nb_active_assets += 1
                print("")
                print("number of active assets =", nb_active_assets)
            except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.DDoSProtection):
                print("Exchange seems not available (maybe too many requests). Will stop now.")
                # exit(-10002)
                os.kill(os.getpid(), 9)
                sys.exit(-999)
                # time.sleep(5)
            except:
                print(sys.exc_info())
                exit(-10003)
    exit(-510)

filter_assets = ""
if args.filter_assets is not None:
    if args.filter_assets.strip() != "":
        filter_assets = args.filter_assets.strip().upper()
        if ("*" in filter_assets and filter_assets.startswith("*") == False and filter_assets.endswith("*") == False) \
                or (
                "*" in filter_assets and filter_assets.startswith("*") == True and filter_assets.endswith("*") == True):
            print(
                "Only one '*' wildcard must be at the start or at the end of the string and not in the middle (not supported).")
            exit(-10004)

retry = args.retry
loop_scan = args.loop
timeframes = args.timeframes
array_tf = set()
x = timeframes.split(',')
for tf in x:
    array_tf.add(tf)
scan_up = args.up
scan_down = args.down

# end of arguments parsing here

debug_delays = False
delay_thread = 0.1  # delay between each start of a thread (in seconds, eg. 0.5 for 500ms, 1 for 1s...)
delay_request = 0  # delay between each request inside of a thread

exchange = None
if args.exchange is not None:
    arg_exchange = args.exchange.lower().strip()
    if arg_exchange in exchanges:
        print(arg_exchange, "is in list")
        exchange = exchanges[arg_exchange]
        # exit(-1)
    else:
        print("This exchange is not supported.")
        exit(-1)
else:
    print("no exchange specified.")
    exit(-2)

delete_results_temp_log(exchange.id)


# exchange = ccxt.binance()
# exchange = ccxt.ftx()

# for tf in exchange.timeframes:
#     print(tf)

# binance.timeframes {'1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h', '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'}
# exchange.set_sandbox_mode(True)

dict_close = {}
dict_open = {}


def get_data_for_timeframe(symbol, tf):
    if tf not in exchange.timeframes:
        print(symbol, tf, "is not in exchange's timeframes.")
        return False
    result = exchange.fetch_ohlcv(symbol, tf, limit=52 + 26 + 26)
    return result


def check_timeframe_up(symbol, tf):
    result = get_data_for_timeframe(symbol, tf)
    if not result:
        return
    # print(tf, symbol, result)
    dframe = pd.DataFrame(result)
    dframe['timestamp'] = pd.to_numeric(dframe[0])
    dframe['open'] = pd.to_numeric(dframe[1])
    dframe['high'] = pd.to_numeric(dframe[2])
    dframe['low'] = pd.to_numeric(dframe[3])
    dframe['close'] = pd.to_numeric(dframe[4])
    dframe['ICH_SSB'] = ta.trend.ichimoku_b(dframe['high'], dframe['low'], window2=26, window3=52).shift(26)
    dframe['ICH_SSA'] = ta.trend.ichimoku_a(dframe['high'], dframe['low'], window1=9, window2=26).shift(26)
    dframe['ICH_KS'] = ta.trend.ichimoku_base_line(dframe['high'], dframe['low'])
    dframe['ICH_TS'] = ta.trend.ichimoku_conversion_line(dframe['high'], dframe['low'])
    dframe['ICH_CS'] = dframe['close'].shift(-26)
    ssb = dframe['ICH_SSB'].iloc[-1]
    ssa = dframe['ICH_SSA'].iloc[-1]
    kijun = dframe['ICH_KS'].iloc[-1]
    tenkan = dframe['ICH_TS'].iloc[-1]
    chikou = dframe['ICH_CS'].iloc[-27]
    price_open = dframe['open'].iloc[-1]
    price_high = dframe['high'].iloc[-1]
    price_low = dframe['low'].iloc[-1]
    price_close = dframe['close'].iloc[-1]
    price_open_chikou = dframe['open'].iloc[-27]
    price_high_chikou = dframe['high'].iloc[-27]
    price_low_chikou = dframe['low'].iloc[-27]
    price_close_chikou = dframe['close'].iloc[-27]
    tenkan_chikou = dframe['ICH_TS'].iloc[-27]
    kijun_chikou = dframe['ICH_KS'].iloc[-27]
    ssa_chikou = dframe['ICH_SSA'].iloc[-27]
    ssb_chikou = dframe['ICH_SSB'].iloc[-27]

    dict_close[symbol] = price_close
    dict_open[symbol] = price_open
    dict_close[symbol] = price_close

    #if price_close > price_open and price_close > kijun and price_close > tenkan and price_close > ssa and price_close > ssb and \
    #    chikou > kijun_chikou and chikou > tenkan_chikou and chikou > ssa_chikou and chikou > ssb_chikou:
    #    return True
    high1 = dframe['high'].iloc[-1]
    high2 = dframe['high'].iloc[-2]
    high3 = dframe['high'].iloc[-3]
    high4 = dframe['high'].iloc[-4]
    high5 = dframe['high'].iloc[-5]
    low1 = dframe['low'].iloc[-1]
    low2 = dframe['low'].iloc[-2]
    low3 = dframe['low'].iloc[-3]
    low4 = dframe['low'].iloc[-4]
    low5 = dframe['low'].iloc[-5]
    #print(symbol, high1, high2, high3, high4)
    # if low2 < low3 < low4:
    #     print(symbol, "2 consecutive lower lows")
    #     return True
    if high2 > high3 and high3 > high4 and high4 < high5:  # and price_close > high2:
        if price_close > ssa and price_close > ssb and price_close > tenkan and price_close > kijun:
            if price_close_chikou > ssa_chikou and price_close_chikou > ssb_chikou and price_close_chikou > tenkan_chikou and price_close_chikou > kijun_chikou:
                # greater_than_all_previous_highs = False
                # for i in range(2, 2):
                #     if price_close > dframe['high'].iloc[-i]:
                #         greater_than_all_previous_highs = True
                #     else:
                #         greater_than_all_previous_highs = False
                #         break
                # if greater_than_all_previous_highs:
                #     return True
                return True

    return False

    # if (ssb > ssa and price_open > ssb and price_close > ssb) or (ssa > ssb and price_open > ssa and price_close > ssa):
    #     pass  # print(symbol, tf, "**** is over the cloud")
    #     if (price_close > kijun):
    #         pass  # print(symbol, tf, "******** is over the kijun")
    #         if (price_close > tenkan):
    #             pass  # print(symbol, tf, "************ is over the tenkan")
    #             if (
    #                     chikou > ssa_chikou and chikou > ssb_chikou and chikou > price_high_chikou and chikou > tenkan_chikou and chikou > kijun_chikou):
    #                 pass  # print(symbol, tf, "**************** has chikou validated")
    #                 if (
    #                         price_close > ssa and price_close > ssb and price_close > tenkan and price_close > kijun and price_close > price_open):
    #                     pass  # print(symbol, tf, "******************** has price validated")
    #                     return True


def check_timeframe_down(symbol, tf):
    result = get_data_for_timeframe(symbol, tf)
    if not result:
        return
    # print(tf, symbol, result)
    dframe = pd.DataFrame(result)
    dframe['timestamp'] = pd.to_numeric(dframe[0])
    dframe['open'] = pd.to_numeric(dframe[1])
    dframe['high'] = pd.to_numeric(dframe[2])
    dframe['low'] = pd.to_numeric(dframe[3])
    dframe['close'] = pd.to_numeric(dframe[4])
    dframe['ICH_SSB'] = ta.trend.ichimoku_b(dframe['high'], dframe['low'], window2=26, window3=52).shift(26)
    dframe['ICH_SSA'] = ta.trend.ichimoku_a(dframe['high'], dframe['low'], window1=9, window2=26).shift(26)
    dframe['ICH_KS'] = ta.trend.ichimoku_base_line(dframe['high'], dframe['low'])
    dframe['ICH_TS'] = ta.trend.ichimoku_conversion_line(dframe['high'], dframe['low'])
    dframe['ICH_CS'] = dframe['close'].shift(-26)
    ssb = dframe['ICH_SSB'].iloc[-1]
    ssa = dframe['ICH_SSA'].iloc[-1]
    kijun = dframe['ICH_KS'].iloc[-1]
    tenkan = dframe['ICH_TS'].iloc[-1]
    chikou = dframe['ICH_CS'].iloc[-27]
    price_open = dframe['open'].iloc[-1]
    price_high = dframe['high'].iloc[-1]
    price_low = dframe['low'].iloc[-1]
    price_close = dframe['close'].iloc[-1]
    price_open_chikou = dframe['open'].iloc[-27]
    price_high_chikou = dframe['high'].iloc[-27]
    price_low_chikou = dframe['low'].iloc[-27]
    price_close_chikou = dframe['close'].iloc[-27]
    tenkan_chikou = dframe['ICH_TS'].iloc[-27]
    kijun_chikou = dframe['ICH_KS'].iloc[-27]
    ssa_chikou = dframe['ICH_SSA'].iloc[-27]
    ssb_chikou = dframe['ICH_SSB'].iloc[-27]

    dict_close[symbol] = price_close
    dict_open[symbol] = price_open
    dict_close[symbol] = price_close

    # if price_close < price_open and price_close < kijun and price_close < tenkan and price_close < ssa and price_close < ssb and \
    #     chikou < kijun_chikou and chikou < tenkan_chikou and chikou < ssa_chikou and chikou < ssb_chikou:
    #     return True

    high1 = dframe['high'].iloc[-1]
    high2 = dframe['high'].iloc[-2]
    high3 = dframe['high'].iloc[-3]
    high4 = dframe['high'].iloc[-4]
    high5 = dframe['high'].iloc[-5]
    low1 = dframe['low'].iloc[-1]
    low2 = dframe['low'].iloc[-2]
    low3 = dframe['low'].iloc[-3]
    low4 = dframe['low'].iloc[-4]
    low5 = dframe['low'].iloc[-5]
    #print(symbol, high1, high2, high3, high4)
    # if low2 < low3 < low4:
    #     print(symbol, "2 consecutive lower lows")
    #     return True
    if low2 < low3 and low3 < low4 and low4 > low5:  # and price_close < low2:
        if price_close < ssa and price_close < ssb and price_close < tenkan and price_close < kijun:
            if price_close_chikou < ssa_chikou and price_close_chikou < ssb_chikou and price_close_chikou < tenkan_chikou and price_close_chikou < kijun_chikou:
                # lower_than_all_previous_lows = False
                # for i in range(2, 2):
                #     if price_close < dframe['low'].iloc[-i]:
                #         lower_than_all_previous_lows = True
                #     else:
                #         lower_than_all_previous_lows = False
                #         break
                # if lower_than_all_previous_lows:
                #     return True
                return True

    return False

    # if (ssb < ssa and price_open < ssb and price_close < ssb) or (ssa < ssb and price_open < ssa and price_close < ssa):
    #     pass  # print(symbol, tf, "**** is over the cloud")
    #     if (price_close < kijun):
    #         pass  # print(symbol, tf, "******** is over the kijun")
    #         if (price_close < tenkan):
    #             pass  # print(symbol, tf, "************ is over the tenkan")
    #             if (
    #                     chikou < ssa_chikou and chikou < ssb_chikou and chikou < price_low_chikou and chikou < tenkan_chikou and chikou < kijun_chikou):
    #                 pass  # print(symbol, tf, "**************** has chikou validated")
    #                 if (
    #                         price_close < ssa and price_close < ssb and price_close < tenkan and price_close < kijun and price_close < price_open):
    #                     pass  # print(symbol, tf, "******************** has price validated")
    #                     return True


dict_results = {}
dict_results_binary = {}
dict_results_evol = {}
highest_percent_evol = 0


def execute_code(symbol, type_of_asset, exchange_id):
    global dict_results, highest_percent_evol
    global ssb, ssa, price_open, price_close, kijun, tenkan, chikou, ssa_chikou, ssb_chikou, price_high_chikou, price_low_chikou
    global tenkan_chikou, kijun_chikou
    # global tweet

    # print(10 * "*", symbol, type_of_asset, exchange.id, 10 * "*")

    binary_result = ""

    # print("Available timeframes : ", exchange.timeframes)
    done = False
    if done == False:  # for tf in exchange.timeframes:

        # if exchange_id in ("binance", "gateio"):
        #     if not symbol.endswith('USDT'):
        #         continue
        # elif exchange_id == "bybit":
        #     # if not symbol == 'BTCPERP':
        #     if not symbol.endswith('PERP'):
        #         continue

        scantype = ""
        if scan_up == False and scan_down == False:
            scantype = "up and down"
        else:
            if scan_up == True:
                scantype += " up "
            if scan_down == True:
                scantype += " down "

        #print(array_tf)

        try:

            # print("scanning", symbol)

            # array_tf = {}
            # if exchange_id == "binance":
            #    array_tf = {"1d", "4h", "1h"}
            # elif exchange_id == "bybit":# and filter_assets == "*PERP":
            #    #array_tf = {"1m", "5m", "15m", "1h"}
            #    array_tf = {"1h"}
            # else:
            #    print("Il faut définir un array de tf pour cet exchange !")

            if "down" in scantype:
                # print("scanning down")
                all_tf_ok = False
                for tf in array_tf:
                    #print("scanning in", tf)
                    if check_timeframe_down(symbol, tf):
                        #beep.beep(3)
                        if exchange_id.upper() == "BYBIT":
                            str_link = "https://tradingview.com/chart/?symbol=" + exchange_id.upper() + ":" + symbol + ".P"
                        elif exchange_id.upper() == "BINANCE":
                            str_link = "https://tradingview.com/chart/?symbol=" + exchange_id.upper() + ":" + symbol
                        else:
                            str_link = "#" + exchange_id.upper()

                        evol = (dict_close[symbol] - dict_open[symbol])/dict_open[symbol]*100
                        str_to_log = "(DOWNTREND) 2 consecutive lower lows for #" + symbol + " in " + tf + " at " + \
                                     str(datetime.now()).split('.')[0] + " " + "price=" + str(dict_close[symbol]) + " " + "{:.2f}".format(evol) + "%"
                        print(str_to_log + " " + str_link)
                        log_to_results(str_to_log + " " + str_link)
                        tweet(str_to_log + "\n" + str_link + " " + "$" + symbol.replace("USDT", "") + "\n" + "#Ichimoku #Crypto #InélidaScanner #BotMonster" + " #" + exchange_id.upper())

            if "up" in scantype:
                # print("scanning up")
                #all_tf_ok = False
                for tf in array_tf:
                    #print("scanning in", tf)
                    if check_timeframe_up(symbol, tf):
                        #beep.beep(3)
                        if exchange_id.upper() == "BYBIT":
                            str_link = "https://tradingview.com/chart/?symbol=" + exchange_id.upper() + ":" + symbol + ".P"
                        elif exchange_id.upper() == "BINANCE":
                            str_link = "https://tradingview.com/chart/?symbol=" + exchange_id.upper() + ":" + symbol
                        else:
                            str_link = "#" + exchange_id.upper()

                        evol = (dict_close[symbol] - dict_open[symbol])/dict_open[symbol]*100
                        str_to_log = "(UPTREND) 2 consecutive higher highs for #" + symbol + " in " + tf + " at " + \
                                     str(datetime.now()).split('.')[0] + " " + "price=" + str(dict_close[symbol]) + " " + "{:.2f}".format(evol) + "%"
                        print(str_to_log + " " + str_link)
                        log_to_results(str_to_log + " " + str_link)
                        tweet(str_to_log + "\n" + str_link + " " + "$" + symbol.replace("USDT", "") + "\n" + "#Ichimoku #Crypto #InélidaScanner #BotMonster" + " #" + exchange_id.upper())
        except:
            #print(symbol, sys.exc_info())
            # print(tf, symbol, sys.exc_info())  # for getting more details remove this line and add line exit(-1) just before the "pass" function
            # log_to_errors(str(datetime.now()) + " " + tf + " " + symbol + " " + str(sys.exc_info()))
            # binary_result += "0"
            pass

        if delay_request > 0:
            # if debug_delays:
            #     print("applying delay_request of", delay_thread, "s after request on timeframe", tf, symbol)
            time.sleep(delay_request)

        done = True


threadLimiter = threading.BoundedSemaphore()


def scan_one(symbol, type_of_asset, exchange_id):
    global threadLimiter
    threadLimiter.acquire()
    try:
        execute_code(symbol, type_of_asset, exchange_id)
    finally:
        threadLimiter.release()


def main_thread():
    maxthreads = 1
    if exchange.id.lower() == "binance":
        maxthreads = 50
        print("setting maxthreads =", maxthreads, "for", exchange.id)
        delay_thread = 0  # 0.1  # delay between each start of a thread (in seconds, eg. 0.5 for 500ms, 1 for 1s...)
        delay_request = 0  # 0.250 # delay between each request inside of a thread
        print("setting delay_thread =", delay_thread, "for", exchange.id)
        print("setting delay_request =", delay_request, "for", exchange.id)
    elif exchange.id.lower() == "ftx":
        maxthreads = 100
        print("setting maxthreads =", maxthreads, "for", exchange.id)
        delay_thread = 0  # delay between each start of a thread (in seconds, eg. 0.5 for 500ms, 1 for 1s...)
        delay_request = 0  # delay between each request inside of a thread
        print("setting delay_thread =", delay_thread, "for", exchange.id)
        print("setting delay_request =", delay_request, "for", exchange.id)
    elif exchange.id.lower() == "gateio":
        maxthreads = 100
        print("setting maxthreads =", maxthreads, "for", exchange.id)
        delay_thread = 0  # delay between each start of a thread (in seconds, eg. 0.5 for 500ms, 1 for 1s...)
        delay_request = 0  # delay between each request inside of a thread
        print("setting delay_thread =", delay_thread, "for", exchange.id)
        print("setting delay_request =", delay_request, "for", exchange.id)
    elif exchange.id.lower() == "bitforex":
        maxthreads = 1
        print("setting maxthreads =", maxthreads, "for", exchange.id)
        delay_thread = 1  # delay between each start of a thread (in seconds, eg. 0.5 for 500ms, 1 for 1s...)
        delay_request = 1  # delay between each request inside of a thread
        print("setting delay_thread =", delay_thread, "for", exchange.id)
        print("setting delay_request =", delay_request, "for", exchange.id)
    else:
        maxthreads = 25
        delay_thread = 0
        delay_request = 0
        print("setting default maxthreads =", maxthreads, "for", exchange.id)
        print("setting default delay_thread =", delay_thread, "for", exchange.id)
        print("setting default delay_request =", delay_request, "for", exchange.id)

    delete_results_log()
    delete_results_evol_log()
    log_to_results("Scan results at : " + str(datetime.now()))

    threadLimiter = threading.BoundedSemaphore(maxthreads)
    # print(threadLimiter)

    ok = False
    while ok is False:
        try:
            markets = exchange.fetch_markets()
            ok = True
            print("markets data obtained successfully")
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.DDoSProtection):
            print("Exchange seems not available (maybe too many requests). Please wait and try again.")
            # exit(-10002)
            if retry is False:
                print("will not retry.")
                exit(-777)
            else:
                print("will retry in 5 sec")
                time.sleep(5)
        except:
            print(sys.exc_info())
            exit(-778)

    threads = []

    # print(markets)
    stop = True
    if loop_scan:
        stop = False

    while True:
        for oneline in markets:
            symbol = oneline['id']
            active = oneline['active']
            type_of_asset = oneline['type']
            #print(symbol, type_of_asset)
            exchange_id = exchange.id.lower()
            base = oneline['base']  # eg. BTC/USDT => base = BTC
            quote = oneline['quote']  # eg. BTC/USDT => quote = USDT
            # print(symbol, "base", base, "quote", quote)

            # print("eval", eval("exchange_id == 'ftx'"))

            # this condition could be commented (and then more assets would be scanned)
            if exchange_id == "ftx":
                if symbol.endswith('HEDGE/USD') or symbol.endswith('CUSDT/USDT') or symbol.endswith('BEAR/USDT') \
                        or symbol.endswith('BEAR/USD') or symbol.endswith('BULL/USDT') or symbol.endswith('BULL/USD') \
                        or symbol.endswith('HALF/USD') or symbol.endswith('HALF/USDT') or symbol.endswith('SHIT/USDT') \
                        or symbol.endswith('SHIT/USD') or symbol.endswith('BEAR2021/USDT') or symbol.endswith(
                    'BEAR2021/USD') \
                        or symbol.endswith('BVOL/USDT') or symbol.endswith('BVOL/USD'):
                    continue

            condition_ok = active and filter_assets in symbol and type_of_asset == "swap"
            if filter_assets.startswith("*"):
                new_filter_assets = filter_assets.replace("*", "")
                new_filter_assets = new_filter_assets.upper()
                condition_ok = active and symbol.endswith(new_filter_assets)
            elif filter_assets.endswith("*"):
                new_filter_assets = filter_assets.replace("*", "")
                new_filter_assets = new_filter_assets.upper()
                condition_ok = active and symbol.startswith(new_filter_assets)

            if condition_ok:  # and ((symbol.endswith("USDT")) or (symbol.endswith("USD"))):  # == symbol: #'BTCUSDT':
                try:
                    t = threading.Thread(target=scan_one, args=(symbol, type_of_asset, exchange_id))
                    threads.append(t)
                    t.start()
                    # print("thread started for", symbol)
                    if delay_thread > 0:
                        if debug_delays:
                            print("applying delay_thread of", delay_thread, "s before next thread start")
                        time.sleep(delay_thread)

                except:
                    pass

        start_time = time.time()

        for tt in threads:
            tt.join()

        end_time = time.time()

        print("--- %s seconds ---" % (end_time - start_time))

        if stop is True:
            break


mainThread = threading.Thread(target=main_thread, args=())
mainThread.start()