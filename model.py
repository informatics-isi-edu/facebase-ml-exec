from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

class ModelManager:
    def __init__(self):
        self.model = self.build_3d_cnn_model()

    def build_3d_cnn_model(self):
        model = Sequential([
            Conv3D(16, (3, 3, 3), activation='relu', input_shape=(256, 256, 256, 1)),
            MaxPooling3D((2, 2, 2)),
            Conv3D(32, (3, 3, 3), activation='relu'),
            MaxPooling3D((2, 2, 2)),
            Conv3D(64, (3, 3, 3), activation='relu'),
            MaxPooling3D((2, 2, 2)),
            Conv3D(128, (3, 3, 3), activation='relu'),
            MaxPooling3D((2, 2, 2)),
            Conv3D(256, (3, 3, 3), activation='relu'),
            MaxPooling3D((2, 2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data, validation_data=None, epochs=10, batch_size=32, callbacks=None):
        self.history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return self.history

    def save_model(self, filepath, format='tf'):
        """Saves the model in the specified format ('tf' or 'h5')."""
        if format == 'h5':
            filepath += '.h5'  # Ensure the filename ends with '.h5'
        self.model.save(filepath)
        print(f"Model saved in {format} format to {filepath}")

    def evaluate(self, test_data):
        test_loss, test_accuracy = self.model.evaluate(test_data)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        return test_loss, test_accuracy

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions
