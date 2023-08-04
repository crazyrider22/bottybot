import pandas as pd
import talib
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from coinbase.wallet.client import Client
import configparser
import logging
import time

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_config(config):
    print("Update Configuration:")
    print("1. API Key")
    print("2. API Secret")
    print("3. Coin ID (e.g., bitcoin)")
    print("4. Currency Pair (e.g., BTC-USD)")
    print("5. Local Currency (e.g., usd)")
    print("0. Return to Main Menu")
    choice = input("Please choose an option to update (0-5): ")

    if choice == '3':
        config['TRADING']['coin_id'] = input("Enter new Coin ID: ")
    elif choice == '4':
        config['TRADING']['currency_pair'] = input("Enter new Currency Pair (e.g., BTC-USD): ")
    elif choice == '5':
        config['TRADING']['local_currency'] = input("Enter new Local Currency (e.g., usd): ")
    # ... (other options)

    # Save updated configuration to file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    print("Configuration updated successfully!")

def get_account_balance(client):
    total_balance = 0.0
    accounts = client.get_accounts()
    for account in accounts['data']:
        balance = float(account['balance']['amount'])
        total_balance += balance
        currency = account['balance']['currency']
        logging.info(f"Balance in {currency}: {balance}")

    logging.info(f"Total balance: {total_balance}")
    return total_balance

def fetch_coingecko_data(coin_id='bitcoin', local_currency='usd', days='30'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': local_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    print(data.head())

def place_buy_order(client, currency_pair, amount):
    buy = client.buy(account_id='primary', amount=amount, currency_pair=currency_pair)
    logging.info(f"Placed buy order: {buy}")

def place_sell_order(client, currency_pair, amount):
    sell = client.sell(account_id='primary', amount=amount, currency_pair=currency_pair)
    logging.info(f"Placed sell order: {sell}")

def live_trading(config=None, backtest=False):
    logging.info("Starting live trading...")
    # Retrieve values from configuration file
    coin_id = config['TRADING']['coin_id']
    local_currency = config['TRADING']['local_currency']
    api_key = config['API']['api_key']
    api_secret = config['API']['api_secret']
    currency_pair = config['TRADING']['currency_pair']
    client = Client(api_key, api_secret)

    # Initialize portfolio
    portfolio = {'cash': get_account_balance(client), 'position': 0}

    # Define the percentage of portfolio to trade
    percentage_to_trade = 50

    # Fetch historical data from CoinGecko
    data = fetch_coingecko_data(coin_id=coin_id, local_currency=local_currency)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms') # Convert timestamp to datetime
    data['close'] = data['price'] # Rename price column to close

    # Calculate technical indicators
    data['Short_EMA'] = talib.EMA(data['close'], timeperiod=10)
    data['Long_EMA'] = talib.EMA(data['close'], timeperiod=50)
    data['EMA_5'] = talib.EMA(data['close'], timeperiod=5)
    data['EMA_20'] = talib.EMA(data['close'], timeperiod=15)
    data['EMA_100'] = talib.EMA(data['close'], timeperiod=100)
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)
    data.dropna(inplace=True)

    # Define the labels
    data['Label'] = (data['close'].shift(-1) > data['close']).astype(int)
    labels = data['Label']

    # Preprocessing for KNN
    features = data[['close', 'Short_EMA', 'Long_EMA', 'RSI', 'EMA_5', 'EMA_20', 'EMA_100']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Splitting data
    XX_train, X_test, y_train, y_test = train_test_split(scaled_features_df, labels, test_size=0.2, random_state=42)

    # Training KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(XX_train, y_train)

    # Continuous loop for live trading
    while True:
        try:
            # Fetch real-time price data
            price_info = client.get_spot_price(currency_pair=currency_pair)
            price = float(price_info['amount'])

            # Calculate short-term and long-term EMA
            data['Short_EMA'] = talib.EMA(data['close'], timeperiod=10)
            data['Long_EMA'] = talib.EMA(data['close'], timeperiod=50)

            # Calculate additional EMAs
            data['EMA_5'] = talib.EMA(data['close'], timeperiod=5)
            data['EMA_20'] = talib.EMA(data['close'], timeperiod=15)
            data['EMA_100'] = talib.EMA(data['close'], timeperiod=100)

            # Calculate RSI
            data['RSI'] = talib.RSI(data['close'], timeperiod=14)

            # Drop rows containing NaN values
            data.dropna(inplace=True)

            # Define the labels
            data['Label'] = (data['close'].shift(-1) > data['close']).astype(int)
            labels = data['Label']

            # Preprocessing for KNN
            features = data[['close', 'Short_EMA', 'Long_EMA', 'RSI', 'EMA_5', 'EMA_20', 'EMA_100']]
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

            # Splitting data
            XX_train, X_test, y_train, y_test = train_test_split(scaled_features_df, labels, test_size=0.2, random_state=42)

            # Training KNN model
            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(XX_train, y_train)

            # Loop through data
            data_to_loop = data.iloc[int(len(data) * 0.8):] if backtest else data
            for i, row in data_to_loop.iterrows():
                try:
                    # Get the current balance (replace with actual function to fetch balance)
                    current_balance = get_account_balance(client)

                    # Calculate the amount to buy or sell based on the percentage
                    amount_to_trade = (current_balance * percentage_to_trade) / 100
                    
                    # Inside the loop
                    current_row = row[['close', 'Short_EMA', 'Long_EMA', 'RSI', 'EMA_5', 'EMA_20', 'EMA_100']].to_frame().T
                    current_features_scaled = scaler.transform(current_row)
                    current_features_df = pd.DataFrame(current_features_scaled, columns=current_row.columns)
                    prediction = knn_model.predict(current_features_df)

                    # Log the current row and prediction
                    logging.info(f"Row {i}: Short_EMA: {row['Short_EMA']}, Long_EMA: {row['Long_EMA']}, RSI: {row['RSI']}, Prediction: {prediction}")

                    # Generate signals with new EMAs
                    buy_signal = (row['Short_EMA'] > row['Long_EMA']) & (row['EMA_5'] > row['EMA_20']) & (row['RSI'] < 50) & (prediction == 1)
                    sell_signal = (row['Short_EMA'] < row['Long_EMA']) & (row['EMA_5'] < row['EMA_20']) & (row['RSI'] > 60) & (prediction == 0)

                    # Log the buy and sell signals
                    logging.info(f"Buy signal: {buy_signal}, Sell signal: {sell_signal}")

                    # Place buy or sell orders if signals are true
                    if buy_signal:
                        place_buy_order(client, currency_pair, amount_to_buy)
                    if sell_signal:
                        place_sell_order(client, currency_pair, amount_to_sell)

                    # Log what's happening
                    logging.info(f"Price: {price}, Other info...")

                   # Sleep for a while before the next iteration
                    time.sleep(60)

                except Exception as e:
                    logging.error(f"Error: {e}")
                    # Handle the error, possibly by waiting and then continuing
                    time.sleep(300)
                    continue


        except Exception as e:
            logging.error(f"Outer loop error: {e}")
            # Handle outer loop errors
            time.sleep(600)
            continue
    
def trading_bot(backtest=False, config=None):
    # Retrieve values from configuration file
    coin_id = config['TRADING']['coin_id']
    local_currency = config['TRADING']['local_currency']
    api_key = config['API']['api_key']
    api_secret = config['API']['api_secret']
    client = Client(api_key, api_secret)
    currency_pair = config['TRADING']['currency_pair']
    
    # Initialize portfolio
    portfolio = {'cash': get_account_balance(client), 'position': 0}

    # Fetch real-time price data from Coinbase
    price_info = client.get_spot_price(currency_pair=currency_pair)
    price = float(price_info['amount'])


    # Fetch historical data from CoinGecko
    data = fetch_coingecko_data(coin_id=coin_id, local_currency=local_currency)

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms') # Convert timestamp to datetime
    data['close'] = data['price'] # Rename price column to close


    # Calculate short-term and long-term EMA
    data['Short_EMA'] = talib.EMA(data['close'], timeperiod=10)
    data['Long_EMA'] = talib.EMA(data['close'], timeperiod=50)

    # Calculate additional EMAs
    data['EMA_5'] = talib.EMA(data['close'], timeperiod=5)
    data['EMA_20'] = talib.EMA(data['close'], timeperiod=15)
    data['EMA_100'] = talib.EMA(data['close'], timeperiod=100)
    
    # Calculate RSI
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)

    # Drop rows containing NaN values
    data.dropna(inplace=True)

    # Define the labels
    data['Label'] = (data['close'].shift(-1) > data['close']).astype(int)
    labels = data['Label']

    # Preprocessing for KNN
    features = data[['close', 'Short_EMA', 'Long_EMA', 'RSI', 'EMA_5', 'EMA_20', 'EMA_100']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Splitting data
    XX_train, X_test, y_train, y_test = train_test_split(scaled_features_df, labels, test_size=0.2, random_state=42)

    # Training KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(XX_train, y_train)

    # Initialize portfolio for backtesting
    starting_balance = 10  # Starting balance in USD
    portfolio = {'cash': starting_balance, 'position': 0} if backtest else None
    
    # Loop through data
    data_to_loop = data.iloc[int(len(data) * 0.8):] if backtest else data
    for i, row in data_to_loop.iterrows():
        try:
            # Inside the loop
            current_row = row[['close', 'Short_EMA', 'Long_EMA', 'RSI', 'EMA_5', 'EMA_20', 'EMA_100']].to_frame().T
            current_features_scaled = scaler.transform(current_row)
            current_features_df = pd.DataFrame(current_features_scaled, columns=current_row.columns)
            prediction = knn_model.predict(current_features_df)

            # Log the current row and prediction
            logging.info(f"Row {i}: Short_EMA: {row['Short_EMA']}, Long_EMA: {row['Long_EMA']}, RSI: {row['RSI']}, Prediction: {prediction}")

            # Generate signals with new EMAs
            buy_signal = (row['Short_EMA'] > row['Long_EMA']) & (row['EMA_5'] > row['EMA_20']) & (row['RSI'] < 50) & (prediction == 1)
            sell_signal = (row['Short_EMA'] < row['Long_EMA']) & (row['EMA_5'] < row['EMA_20']) & (row['RSI'] > 60) & (prediction == 0)

            # Log the buy and sell signals
            logging.info(f"Buy signal: {buy_signal}, Sell signal: {sell_signal}")

            if backtest:
                # Simulate trading for backtesting
                if buy_signal:
                    amount_to_buy = 1
                    price = row['close']
                    portfolio['cash'] -= price * amount_to_buy
                    portfolio['position'] += amount_to_buy
                    logging.info(f"Buying {amount_to_buy} at {price}. Current portfolio: {portfolio}")
                elif sell_signal:
                    amount_to_sell = 1
                    price = row['close']
                    portfolio['cash'] += price * amount_to_sell
                    portfolio['position'] -= amount_to_sell
                    logging.info(f"Selling {amount_to_sell} at {price}. Current portfolio: {portfolio}")

        except Exception as e:
            logging.error(f"Error processing row {i}: {e}")
            continue

    # Calculate performance metrics for backtesting
    if backtest:
        ending_balance = portfolio['cash'] + portfolio['position'] * data['close'].iloc[-1]
        returns = (ending_balance / starting_balance) - 1
        logging.info(f"Starting balance: ${starting_balance}")
        logging.info(f"Ending balance: ${ending_balance}")
        logging.info(f"Total returns: {returns * 100}%")

def view_config(config):
    print("Current Configuration:")
    for section in config.sections():
        print(f"[{section}]")
        for key, value in config.items(section):
            print(f"{key} = {value}")
    print()

def main():
    # Read configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Authenticate with Coinbase API
    api_key = config['API']['api_key']
    api_secret = config['API']['api_secret']
    client = Client(api_key, api_secret)  # Instantiate the client here
    
    while True:
        print("Welcome to the Crypto Trading Bot!")
        print("1. Start Live Trading")
        print("2. Run Backtest")
        print("3. View Configuration")  # New option to view configuration
        print("4. Update Configuration")
        print("5. Check Account Balance")  # New option
        print("6. Exit")  # Updated exit option number
        choice = input("Please choose an option (1-5): ")

        if choice == '1':
            print("Starting live trading...")
            live_trading(config)
        elif choice == '2':
            print("Running backtest...")
            trading_bot(backtest=True, config=config)
        elif choice == '3':  # New choice to view configuration
            view_config(config)
        elif choice == '4':
            update_config(config)
        elif choice == '5':
            print("Checking account balance...")
            balance = get_account_balance(client)
            print(f"Total balance: ${balance}")
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
