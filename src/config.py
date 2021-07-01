# storing API data
api_key = ""
api_secret = ""
paper_api_key = ""
paper_api_secret = ""
paper_endpoint = "https://paper-api.alpaca.markets"
endpoint = "https://api.alpaca.markets"

polygon_api_key = ""

APCA_API_KEY_ID = paper_api_key # Your API Key
APCA_API_SECRET_KEY = paper_api_secret # Your API Secret Key
APCA_API_BASE_URL = paper_endpoint # (for live) Specify the URL for API calls, Default is live, you must specify https://paper-api.alpaca.markets to switch to paper endpoint!
APCA_API_DATA_URL = "https://data.alpaca.markets" # Endpoint for data API
APCA_RETRY_MAX = 3 # 3 The number of subsequent API calls to retry on timeouts
APCA_RETRY_WAIT = 3 # 3 seconds to wait between each retry attempt
APCA_RETRY_CODES= 429, 504 # comma-separated HTTP status code for which retry is attempted
DATA_PROXY_WS = "" # When using the alpaca-proxy-agent you need to set this environment variable as described here