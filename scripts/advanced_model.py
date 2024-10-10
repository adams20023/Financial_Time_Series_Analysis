import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

def build_advanced_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = Bidirectional(LSTM(100, return_sequences=True))(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = Bidirectional(LSTM(100, return_sequences=True))(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Attention Mechanism
    attention_data = Dense(1, activation='tanh')(lstm_out)
    attention_weights = Dense(1, activation='softmax')(attention_data)
    context_vector = attention_weights * lstm_out
    context_vector = K.sum(context_vector, axis=1)  # Replacing np.sum with Keras sum function
    
    outputs = Dense(1)(context_vector)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_advanced_model():
    # Load preprocessed data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # Build model
    model = build_advanced_model((X_train.shape[1], 1))
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('models/best_advanced_model.keras', save_best_only=True)  # Updated file extension
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Save final model
    model.save('models/final_advanced_model.keras')  # Updated file extension
    
    # Save training history
    np.save('models/advanced_history.npy', history.history)
    
    print('Advanced model training completed.')

if __name__ == "__main__":
    train_advanced_model()

