import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import ccxt


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialisation du modèle
input_size = 1
hidden_size = 50
output_size = 1

model = LSTMModel(input_size, hidden_size, output_size)
scaler = MinMaxScaler(feature_range=(0, 1))

# Paramètres
time_window = 60
data_buffer = []

def process_realtime_data(current_price):
    global data_buffer
    data_buffer.append(current_price)
    
    # Normalisation dynamique
    if len(data_buffer) > time_window:
        data_buffer.pop(0)
    scaled_data = scaler.fit_transform(np.array(data_buffer).reshape(-1, 1))
    
    # Créer un tenseur pour le modèle
    X = torch.FloatTensor(scaled_data[-time_window:]).unsqueeze(0)
    with torch.no_grad():
        prediction = model(X)
    predicted_price = scaler.inverse_transform(prediction.numpy().reshape(-1, 1))[0][0]
    
    # Mise à jour du modèle (entraînement incrémentiel)
    if len(data_buffer) > time_window + 1:
        X_train = torch.FloatTensor(scaled_data[-time_window-1:-1]).unsqueeze(0)
        y_train = torch.FloatTensor(scaled_data[-1].reshape(-1, 1))
        train_step(X_train, y_train)
    
    return predicted_price



def get_latest_bitcoin_price():
    try:
        exchange = ccxt.binance()  # Utilisation de Binance comme exchange
        symbol = 'BTC/USDT'  # Paire de trading Bitcoin/USDT
        ticker = exchange.fetch_ticker(symbol)  # Récupération des données du ticker
        latest_price = ticker['last']  # Extraction du dernier prix
        return latest_price
    except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.DDoSProtection) as e:
        print(f"Erreur lors de la récupération du prix du Bitcoin: {str(e)}")
        return None

if __name__ == "__main__":
    while(True):
        price = get_latest_bitcoin_price()
        if price:
            print(f"Le dernier prix du Bitcoin est : {price} USDT")
            process_realtime_data(price)