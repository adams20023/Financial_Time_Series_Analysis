import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def build_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))  # Add dropout for regularization
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.2))  # Add another dropout layer
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def plot_loss(history):
    # Plot training and validation loss over epochs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_model():
    # Load preprocessed data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # Build model
    model = build_model((X_train.shape[1], 1))
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # Reduce patience for early stopping
    model_checkpoint = ModelCheckpoint('models/best_model.keras', save_best_only=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Save final model
    model.save('models/final_model.keras')
    
    # Save training history
    np.save('models/history.npy', history.history)
    
    # Plot training history
    plot_loss(history)
    
    print('Model training completed.')

if __name__ == "__main__":
    train_model()

