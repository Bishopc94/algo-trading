# with help from https://github.com/alpacahq/alpaca-trade-api-python
# https://github.com/alpacahq

import time
import alpaca_trade_api as tradeapi
from polygon import RESTClient
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, polygon_api_key, APCA_API_DATA_URL

def getAccountData():
    api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL)

    # Get our account information.
    account = api.get_account()

    # Check if our account is restricted from trading.
    if account.trading_blocked:
        print('Account is currently restricted from trading.')

    # Check how much money we can use to open new positions.
    print('${} is available as buying power.'.format(account.buying_power))

def my_custom_process_message(message):
    print("this is my custom message processing", message)

def my_custom_error_handler(ws, error):
    print("this is my custom error handler", error)

def main():
    getAccountData()

    # RESTClient can be used as a context manager to facilitate closing the underlying http session
    # https://requests.readthedocs.io/en/master/user/advanced/#session-objects
    with RESTClient(polygon_api_key) as client:
        resp = client.stocks_equities_daily_open_close("AAPL", "2021-06-11")
        print(f"On: {resp.from_} Apple opened at {resp.open} and closed at {resp.close}")


if __name__ == '__main__':
    main()