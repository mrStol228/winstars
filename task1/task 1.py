import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100, random_state=52):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_reshaped, y_train)
        if X_val is not None and y_val is not None:
            X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
            accuracy = self.model.score(X_val_reshaped, y_val)
            print(f"[Random Forest] Validation accuracy: {accuracy:.4f}")

    def predict(self, X):
        X_reshaped = X.reshape(X.shape[0], -1)
        return self.model.predict(X_reshaped)

class NeuralNetworkMnistClassifier(MnistClassifierInterface):
    def __init__(self, input_shape=(784,), num_classes=10, epochs=5, batch_size=128):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=self.input_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
        y_train_cat = to_categorical(y_train, self.num_classes)
        if X_val is not None and y_val is not None:
            X_val_reshaped = X_val.reshape(X_val.shape[0], -1).astype('float32') / 255.0
            y_val_cat = to_categorical(y_val, self.num_classes)
            self.model.fit(X_train_reshaped, y_train_cat, validation_data=(X_val_reshaped, y_val_cat), epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        else:
            self.model.fit(X_train_reshaped, y_train_cat, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, X):
        X_reshaped = X.reshape(X.shape[0], -1).astype('float32') / 255.0
        preds_proba = self.model.predict(X_reshaped)
        return np.argmax(preds_proba, axis=1)

class CNNMnistClassifier(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, epochs=5, batch_size=128):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if len(X_train.shape) < 4:
            X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_train = X_train.astype('float32') / 255.0
        y_train_cat = to_categorical(y_train, self.num_classes)
        if X_val is not None and y_val is not None:
            if len(X_val.shape) < 4:
                X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
            X_val = X_val.astype('float32') / 255.0
            y_val_cat = to_categorical(y_val, self.num_classes)
            self.model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        else:
            self.model.fit(X_train, y_train_cat, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, X):
        if len(X.shape) < 4:
            X = X.reshape(X.shape[0], 28, 28, 1)
        X = X.astype('float32') / 255.0
        preds_proba = self.model.predict(X)
        return np.argmax(preds_proba, axis=1)

class MnistClassifier:
    def __init__(self, algorithm='cnn', **kwargs):
        self.algorithm = algorithm.lower()
        self.model = None
        if self.algorithm == 'rf':
            self.model = RandomForestMnistClassifier(**kwargs)
        elif self.algorithm == 'nn':
            self.model = NeuralNetworkMnistClassifier(**kwargs)
        elif self.algorithm == 'cnn':
            self.model = CNNMnistClassifier(**kwargs)
        else:
            raise ValueError("Unknown algorithm! Choose from 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.train(X_train, y_train, X_val, y_val)

    def predict(self, X):
        return self.model.predict(X)

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_val = X_train[-5000:]
    y_val = y_train[-5000:]
    X_train = X_train[:-5000]
    y_train = y_train[:-5000]
    rf = MnistClassifier(algorithm='rf', n_estimators=50, random_state=42)
    print("Training Random Forest...")
    rf.train(X_train, y_train, X_val, y_val)
    preds_rf = rf.predict(X_test)
    acc_rf = np.mean(preds_rf == y_test)
    print(f"Random Forest test accuracy: {acc_rf:.4f}\n")
    nn = MnistClassifier(algorithm='nn', epochs=3, batch_size=128)
    print("Training Feed-Forward NN...")
    nn.train(X_train, y_train, X_val, y_val)
    preds_nn = nn.predict(X_test)
    acc_nn = np.mean(preds_nn == y_test)
    print(f"NN test accuracy: {acc_nn:.4f}\n")
    cnn = MnistClassifier(algorithm='cnn', epochs=3, batch_size=128)
    print("Training CNN...")
    cnn.train(X_train, y_train, X_val, y_val)
    preds_cnn = cnn.predict(X_test)
    acc_cnn = np.mean(preds_cnn == y_test)
    print(f"CNN test accuracy: {acc_cnn:.4f}\n")

if __name__ == "__main__":
    main()
