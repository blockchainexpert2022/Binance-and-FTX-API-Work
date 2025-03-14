import ccxt
import sys
import os
import argparse
import time
from datetime import datetime
import pytz
import re

def fetch_markets(exchange, filter_assets):
    try:
        markets = exchange.fetch_markets()
        pattern = re.compile(filter_assets.replace('*', '.*'), re.IGNORECASE)
        symbols = [market['symbol'] for market in markets if pattern.match(market['symbol']) and market['spot']]
        return symbols
    except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.DDoSProtection) as e:
        print(f"Exchange error: {str(e)}")
        os.kill(os.getpid(), 9)
        sys.exit(-999)

def fetch_ohlcv(exchange, symbol, timeframe, limit):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        print(f"Error fetching OHLCV data for {symbol}: {str(e)}")
        return []

def is_evening_star(candles):
    if len(candles) != 3:
        return False

    first_candle, second_candle, third_candle = candles

    first_open = first_candle[1]
    first_close = first_candle[4]
    second_open = second_candle[1]
    second_close = second_candle[4]
    third_open = third_candle[1]
    third_close = third_candle[4]

    # First candle: large bullish
    if first_close <= first_open:
        return False

    # Second candle: small body
    if abs(second_close - second_open) >= abs(first_open - first_close) / 2:
        return False

    # Third candle: large bearish
    if third_close >= third_open or abs(third_close - third_open) <= abs(first_open - first_close) / 2:
        return False

    # Third candle closes well within the body of the first candle
    if third_close >= first_open or third_close <= first_close:
        return False

    return True

def is_tradable(exchange, symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker.get('ask') is not None
    except Exception as e:
        print(f"Error checking tradability for {symbol}: {str(e)}")
        return False

def convert_to_tradingview_symbol(ccxt_symbol):
    base, quote = ccxt_symbol.split('/')
    return f"BINANCE:{base}{quote}"

def scan_and_display_assets(exchange, symbols, timeframe, output_file):
    paris_tz = pytz.timezone('Europe/Paris')
    current_time = datetime.now(paris_tz).strftime('%Y-%m-%d %H:%M:%S')

    header = f"Scan results at {current_time} (Paris time):\n"
    print(header.strip())

    limit = 3  # We need the last three candles to check for Evening Star pattern
    for symbol in symbols:
        if not is_tradable(exchange, symbol):
            continue
        
        ohlcv = fetch_ohlcv(exchange, symbol, timeframe, limit)
        if len(ohlcv) >= 3:
            candles = ohlcv[-3:]  # Last three candles

            if is_evening_star(candles):
                current_candle = candles[-1]
                current_open = current_candle[1]
                current_close = current_candle[4]
                current_candle_color = "green" if current_close > current_open else "red"

                evolution = ((current_close - current_open) / current_open) * 100

                result = (
                    f"[{current_time}] {symbol} ({convert_to_tradingview_symbol(symbol)}) "
                    f"Previous pattern: Evening Star, "
                    f"Current price: {current_close:.4f}, Current color: {current_candle_color}, "
                    f"Price evolution: {evolution:.2f}%\n"
                )
                print(result.strip())
                with open(output_file, 'a') as f:
                    f.write(result)
    
    print("Scan complete.")

def main():
    parser = argparse.ArgumentParser(description='Scan assets for the Evening Star pattern in the previous candlesticks.')
    parser.add_argument('--interval', type=int, default=600, help='Interval between scans in seconds.')
    parser.add_argument('--output', type=str, default='scan_results.txt', help='Output file for results.')
    parser.add_argument('--filter', type=str, default='*USDT', help='Filter pattern for symbols to scan (e.g., *USDT to scan only USDT pairs).')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe to scan (e.g., 1m, 5m, 15m, 1h).')
    parser.add_argument('--list-timeframes', action='store_true', help='List available timeframes and exit.')
    args = parser.parse_args()

    exchange_name = "binance"  # Change to your exchange
    filter_pattern = args.filter
    timeframe = args.timeframe
    interval = args.interval
    output_file = args.output

    exchange = getattr(ccxt, exchange_name)()

    if args.list_timeframes:
        print("Available timeframes:", ', '.join(exchange.timeframes))
        sys.exit(0)

    print("Fetching markets...")
    symbols = fetch_markets(exchange, filter_pattern)
    if not symbols:
        print(f"No symbols found for filter pattern {filter_pattern} in the spot market. Exiting.")
        sys.exit(-1)

    print(f"Number of symbols to be tracked: {len(symbols)}")

    while True:
        scan_and_display_assets(exchange, symbols, timeframe, output_file)
        print(f"Waiting for {interval} seconds before the next scan...")
        time.sleep(interval)

if __name__ == "__main__":
    main()
