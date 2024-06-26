import sys

import ccxt
import json
from pprint import pprint

from ccxt import binance, Exchange, InsufficientFunds, InvalidOrder

print('CCXT Version:', ccxt.__version__)

exchange = ccxt.binance({
    'apiKey': '',
    'secret': '',
    'enableRateLimit': True,  # https://github.com/ccxt/ccxt/wiki/Manual#rate-limit
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    },
})


#exchange.set_sandbox_mode(True)  # comment if you're not using the testnet
markets = exchange.load_markets()
exchange.verbose = False  # debug output


def get_all_balances():
    balance = exchange.fetch_balance()
    # pprint(balance)

    total = 0.0
    for k, v in balance.items():
        #print(k, v, type(v))
        if type(v) is dict:
            for kk, vv in v.items():
                if kk == "free":
                    if float(vv) > 0:
                        print("get_all_balances(1):", kk + " ", k, "{:.16f}".format(vv))
                if kk == "total":
                    if float(vv) > 0:
                        if k != "USDT":
                            try:
                                buy, sell = get_ticker(k + "/USDT")
                                #print("sell", sell, "equivalent in USDT", sell * float(vv))
                                print("get_all_balances(2):", kk, k, "{:.16f}".format(vv), "equivalent in USDT", sell * float(vv))
                                total = total + sell * float(vv)
                            except:
                                pass

    print("get_all_balances(3): total equivalent in usdt (relative to sell prices)", total)
    buy_eur, sell_eur = get_ticker("EUR/USDT")
    total_euro = total / sell_eur
    print("get_all_balances(3): total equivalent in eur (relative to sell price)", total_euro)
    usdt_balance = get_usdt_balance()
    total_wallet_usdt = total + usdt_balance
    print("get_all_balances(3): total wallet in usdt (relative to sell price)", total_wallet_usdt)
    total_wallet_euro = (total + usdt_balance) / sell_eur
    print("get_all_balances(3): total wallet in eur (relative to sell price)", total_wallet_euro)

    print("get_all_balances(4): ", end=" ")
    for i in balance.items():
        # print(i)
        # print("i[0]", i[0])
        # print("i[1]", i[1])
        if i[0] == 'free':
            print(i[1])
            # usdt = (i[1]['USDT'])


def get_usdt_balance():
    balance = exchange.fetch_balance()
    # pprint(balance)

    usdt = 0.0

    for i in balance.items():
        # print(i)
        # print("i[0]", i[0])
        # print("i[1]", i[1])
        if i[0] == 'free':
            #print(i[1])
            usdt = (i[1]['USDT'])

    return usdt


# eg. I want to know how much BTC I have in my wallet
def get_balance_of(crypto_to_get):  # eg. get_balance_of("BTC"), get_balance_of("ETH"), get_balance_of("XRP")...
    balance = exchange.fetch_balance()
    # pprint(balance)

    balance_of_crypto_to_get = 0.0

    for i in balance.items():
        # print(i)
        # print("i[0]", i[0])
        # print("i[1]", i[1])
        if i[0] == 'free':
            #print("i[1]", i[1])
            if crypto_to_get in i[1]:
                #print("OK : crypto to sell has been found in the list of available cryptos from server i[1]")
                balance_of_crypto_to_get = (i[1][crypto_to_get])
                break
            balance_of_crypto_to_get = -1.0

    if balance_of_crypto_to_get == -1.0:
        print("get_balance_of: ERROR : crypto to sell has not been found in the list of available cryptos from server")
        return balance_of_crypto_to_get

    print("get_balance_of:", crypto_to_get, balance_of_crypto_to_get)
    return balance_of_crypto_to_get


# eg. I want to sell 1 BTC
def sell(crypto_to_sell, crypto_to_get, quantity_of_crypto_to_sell):  # eg. sell("BTC", "USDT", 1.5), sell("XRP", "USDT", 25)...
    symbol_to_trade = crypto_to_sell + "/" + crypto_to_get
    type = 'market'  # or 'market'
    side = 'sell'  # or 'buy'
    amount_to_sell = "{:.16f}".format(quantity_of_crypto_to_sell)   # todo : check if 16 digits after point is ok ?
    price = None  # or None
    # extra params and overrides if needed
    params = {
        'test': False,  # test if it's valid, but don't actually place it
    }

    print("sell: Before order sending")
    try:
        order = exchange.create_order(symbol_to_trade, type, side, amount_to_sell, price, params)
        print("sell: SELL Order sent, here are the details:")
        print(order)
        return 0
    except InvalidOrder:
        print("sell: exception", sys.exc_info())
        return -1
    except:
        print("sell: exception", sys.exc_info())
        return -2


# eg. I want to buy 1 BTC and pay in USDT. Returns the executed quantity (effective quantity bought)
def buy(crypto_to_buy, crypto_for_payment, quantity_of_crypto_to_buy):  # eg. buy("BTC", "USDT", 2.5), buy("XRP", "USDT", 100)...
    symbol = crypto_to_buy + "/" + crypto_for_payment
    type = 'market'  # or 'market'
    side = 'buy'  # or 'buy'
    amount = "{:.16f}".format(quantity_of_crypto_to_buy)    # todo : check if 16 digits after point is ok ?
    price = None  # or None
    # extra params and overrides if needed
    params = {
        'test': False,  # test if it's valid, but don't actually place it
    }

    print("buy: Before order sending")
    executed_quantity = 0.0
    try:
        order = exchange.create_order(symbol, type, side, amount, price, params)

        print("buy: BUY Order sent, here are the details:")
        print(order)

        for item in order.items():
            #print("item", item)
            for subitem in item:
                try:
                    #print("item", item, "subitem", subitem)
                    for subitem2 in subitem:
                        if subitem2 == "executedQty":
                            #print("item", item, "subitem", subitem, "subitem2", subitem2)
                            #print(subitem["executedQty"])
                            executed_quantity = float(subitem["executedQty"])
                            #break
                except:
                    pass

        print("buy: Executed order quantity", executed_quantity)
        return executed_quantity

    except InvalidOrder:
        print("buy: exception", sys.exc_info())
        return -1
    except:
        print("buy: exception", sys.exc_info())
        return -2


# eg. I want to buy BTC for a specified amount of USDT
def buy_for_amount_of(crypto_to_buy, crypto_for_payment, amount_of_crypto_for_payment):  # eg. buy("BTC", "USDT", 50) : That will try to buy BTC for 50 usdt
    print("buy_for_amount_of: Getting price for ", crypto_to_buy, "/", crypto_for_payment)
    ticker = exchange.fetch_ticker(crypto_to_buy + "/" + crypto_for_payment)
    print("buy_for_amount_of: ticker", ticker)
    print("buy_for_amount_of: ", ticker["symbol"], "sell price", ticker["bid"], "buy price", ticker["ask"], "close price", ticker["close"])
    crypto_price = ticker["ask"]
    print("buy_for_amount_of: Buy price for ", crypto_to_buy, "/", crypto_for_payment, crypto_price)
    quantity_of_crypto_to_buy = amount_of_crypto_for_payment / crypto_price
    print("buy_for_amount_of: Quantity of ", crypto_to_buy, "/", crypto_for_payment,  "to buy for ", amount_of_crypto_for_payment, "usdt", "=", "{:.16f}".format(quantity_of_crypto_to_buy))
    symbol = crypto_to_buy + "/USDT"
    type = 'market'  # or 'market'
    side = 'buy'  # or 'buy'
    amount = "{:.16f}".format(quantity_of_crypto_to_buy)  # todo : check if 16 digits after point is ok ?
    price = None  # or None
    # extra params and overrides if needed
    params = {
        'test': False,  # test if it's valid, but don't actually place it
    }

    try:
        print("buy_for_amount_of: Before order sending")
        order = exchange.create_order(symbol, type, side, amount, price, params)
        print("buy_for_amount_of: BUY Order sent, here are the details:")
        print(order)
        return ""
    except InsufficientFunds:
        #print("exception", sys.exc_info())
        print("buy_for_amount_of: Fonds insuffisants.")
        return "Insufficient funds"
    except InvalidOrder:
        print("buy_for_amount_of: exception", sys.exc_info())


# eg I want to convert all my ETH to USDT : sell_all_crypto_for("ETH", "USDT")
def sell_all_crypto_for(crypto_to_sell, crypto_to_get):
    while get_balance_of(crypto_to_sell) > 0:
        sell(crypto_to_sell, crypto_to_get, get_balance_of(crypto_to_sell))

#def buy_crypto_until_quantity_reached(crypto_to_buy, quantity_to_reach):
#    while get_balance_of(crypto_to_buy < quantity_to_reach):


# eg. I want to know what is the minimum allowed for buying BTC/USDT (eg. buying is allowed for a minimum of 10 usdt)
def get_allowed_min_notional(crypto_to_buy, crypto_for_payment):
    print("get_allowed_min_notional: Current market items")
    print("get_allowed_min_notional: Searching if ", crypto_to_buy, "/", crypto_for_payment, " is available for trading")
    symbol_found = False
    # print(exchange.markets.items())
    for line in exchange.markets.items():
        #print("get_allowed_minimum_to_buy: line", line)  # décommenter pour voir les différents assets tradables
        if line[0] == crypto_to_buy + "/" + crypto_for_payment:
            print(crypto_to_buy + "/" + crypto_for_payment, "found (available for trading)")
            symbol_found = True
            print("get_allowed_min_notional: line[1]", line[1]["info"]["filters"])
            for subline in line[1]["info"]["filters"]:
                if subline["filterType"] == "MIN_NOTIONAL":
                    print("get_allowed_min_notional: minimum allowed to buy in", crypto_for_payment, "=", subline["minNotional"])
                    return float(subline["minNotional"])
            break
    if symbol_found is False:
        return -1


# eg. I want to know what is the maximum quantity allowed to be sold for a crypto in a single market order
def get_allowed_market_lot_size(crypto_to_buy, crypto_for_payment):
    print("get_allowed_market_lot_size: Current market items")
    print("get_allowed_market_lot_size: Searching if ", crypto_to_buy, "/", crypto_for_payment, " is available for trading")
    symbol_found = False
    # print(exchange.markets.items())
    for line in exchange.markets.items():
        #print("get_allowed_minimum_to_buy: line", line)  # décommenter pour voir les différents assets tradables
        if line[0] == crypto_to_buy + "/" + crypto_for_payment:
            print(crypto_to_buy + "/" + crypto_for_payment, "found (available for trading)")
            symbol_found = True
            print("get_allowed_market_lot_size: line[1]", line[1]["info"]["filters"])
            for subline in line[1]["info"]["filters"]:
                if subline["filterType"] == "MARKET_LOT_SIZE":
                    print("get_allowed_market_lot_size: maximum allowed to sell in", crypto_for_payment, "=", subline["maxQty"])
                    return float(subline["maxQty"])
            break
    if symbol_found is False:
        return -1


# eg. I want to know if "BTC/USDT" is available for trading : is_tradable("BTC/USDT")
def is_tradable(symbol_to_trade):
    print("is_tradable: Current market items")
    print("is_tradable: Searching if ", symbol_to_trade, " is available for trading")
    symbol_found = False
    # print(exchange.markets.items())
    for line in exchange.markets.items():
        #print("get_allowed_minimum_to_buy: line", line)  # décommenter pour voir les différents assets tradables
        if line[0] == symbol_to_trade:
            symbol_found = True
            break
    return symbol_found


def get_tradable_pairs():
    array_pairs = []
    print("get_tradable_pairs: tradable pairs:")
    for line in exchange.markets.items():
        print(line[0], end=" ")
        array_pairs.append(line[0])
    print("")
    return array_pairs


# eg. I want to know the current prices for XRP/USDT : get_ticker("XRP/USDT")
def get_ticker(symbol_to_get):
    ticker = exchange.fetch_ticker(symbol_to_get)
    # print(ticker)
    # print("get_ticker:", ticker)
    # print(ticker["symbol"])

    # print(ticker["symbol"], "sell price", ticker["bid"], "buy price", ticker["ask"], "close price", ticker["close"])
    sell_price = float(ticker["bid"])
    buy_price = float(ticker["ask"])
    # print("get_ticker: buy price", buy_price, "sell_price", sell_price)
    return buy_price, sell_price


# eg. I want to sell any "/USDT" pair to get USDT (I want to empty every "/USDT" pair)
def sell_all_usdt_pairs():
    array_tradable_pairs = get_tradable_pairs()
    for pair_item in array_tradable_pairs:
        pair = str(pair_item)
        if pair.endswith("/USDT"):
            crypto = pair.replace("/USDT", "")
            balance = get_balance_of(crypto)
            print("sell_all_usdt_pairs:", crypto, get_balance_of(crypto))
            if balance > 0:
                allowed_maximum_to_sell = get_allowed_market_lot_size(crypto, "USDT")
                print("sell_all_usdt_pairs: Allowed maximum to sell=", allowed_maximum_to_sell)
                while get_balance_of(crypto) > allowed_maximum_to_sell:
                    sell(crypto, "USDT", allowed_maximum_to_sell)
                while get_balance_of(crypto) > 0:
                    result = sell(crypto, "USDT", get_balance_of(crypto))
                    if result != 0:
                        break


def buy_all_usdt_pairs(amount_in_usdt_for_each):
    array_tradable_pairs = get_tradable_pairs()
    for pair_item in array_tradable_pairs:
        pair = str(pair_item)
        if pair.endswith("/USDT"):
            crypto = pair.replace("/USDT", "")
            balance = get_balance_of(crypto)
            print("buy_all_usdt_pairs:", crypto, balance)
            buy_for_amount_of(crypto, "USDT", amount_in_usdt_for_each)

            # max_lot_size = get_allowed_market_lot_size(crypto, "USDT")
            # while get_balance_of(crypto) > max_lot_size:
            #     buy_for_amount_of(crypto, "USDT", max_lot_size)
            # balance = get_balance_of(crypto)
            # if balance > 0:
            #     buy_for_amount_of(crypto, "USDT", get_balance_of(crypto))


initial_usdt_balance = get_usdt_balance()
print("main: Current balance in USDT", initial_usdt_balance)

print("main: Current market items")
# print("main: Searching if BTC/USDT is available for trading")
# btcusdt_found = False
# # print(exchange.markets.items())
# print("main: tradable pairs:")
# for line in exchange.markets.items():
#     print(line[0], end=" ")

print("")
#sell("BTC", "USDT", get_balance_of("BTC"))

# btc_balance = get_balance_of("BTC")
# if btc_balance > 0:
#     sell("BTC/USDT", btc_balance)
# else:
#     print("KO : balance = 0, nothing to sell.")

# print("usdt balance", get_balance_of("USDT"))
# balance = get_balance_of("USDT")
# buy_for_usdt("BTC", balance)

# buy_for_usdt("BTC", get_balance_of("USDT") )
# sell("BTC", "USDT", get_balance_of("BTC"))

# print("MIN ALLOWED TO BUY IN USDT: ", get_allowed_minimum_to_buy("BTC", "USDT"))
# sell("BTC", "USDT", get_balance_of("BTC"))

#sell("BTC", "USDT", get_balance_of("BTC"))
#buy_for_amount_of("BTC", "USDT", 835)

# print(get_allowed_minimum_to_buy("ETH", "USDT"))
#buy_for_amount_of("ETH", "USDT", 1000)
#buy("ETH", "USDT", 40)

# while get_balance_of("ETH") > 0:
#      sell("ETH", "USDT", get_balance_of("ETH"))
#
# while get_balance_of("ETH") < 50:
#     buy("ETH", "USDT", 50-get_balance_of("ETH"))

#get_allowed_minimum_to_buy("ETH", "USDT")

#get_allowed_minimum_to_buy("XRP", "USDT")

#exit(-2)

#buy_for_amount_of("TRX", "USDT", 1000)
#buy_all_usdt_pairs(1000)

#sell_all_usdt_pairs()

# sell("LAZIO", "USDT", get_balance_of("LAZIO"))
# sell("PSG", "USDT", get_balance_of("PSG"))
# sell("ACM", "USDT", get_balance_of("ACM"))
# sell("CITY", "USDT", get_balance_of("CITY"))
get_all_balances()

#sell_all_crypto_for("ETH", "USDT")
#effective_quantity_bought = buy("ETH", "USDT", 50)
#print(effective_quantity_bought, "has been bought")

# effective_quantity_bought = 0
# while effective_quantity_bought < 50:
#     effective_quantity_bought = effective_quantity_bought + buy("ETH", "USDT", 50 - get_balance_of("ETH"))

# effective_quantity_bought = 0
# while effective_quantity_bought < 1:
#     effective_quantity_bought = effective_quantity_bought + buy("BTC", "USDT", 1 - get_balance_of("BTC"))





#buy("XRP", "USDT", 100)

#get_all_balances()

exit(-2)

