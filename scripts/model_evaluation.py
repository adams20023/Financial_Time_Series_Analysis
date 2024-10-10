import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def evaluate_model():
    # Load data
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # Load scaler
    scaler = np.load('models/scaler.npy', allow_pickle=True).item()
    
    # Load model (Update file extension to .keras)
    model = load_model('models/best_model.keras')
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Plot results
    plt.figure(figsize=(14,5))
    plt.plot(y_test_scaled, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('outputs/stock_price_prediction.png')
    plt.show()
    
    print('Model evaluation completed.')

if __name__ == "__main__":
    evaluate_model()

