import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import KFold


def load_dataset():
    # Loading the dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # Reshaping data to have a single color channel since images are grayscale
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    # Also using one hot encoding with 1 for the index of the class, and 0 for everything else
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return X_train, Y_train, X_test, Y_test

def preprocess_pixels(train, test):
    # Converting to float
    train = train.astype('float32')
    test = test.astype('float32')
    # Normalising it
    train = train / 255.0
    test = test / 255.0

    return train, test

def define_model():
    # defining the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # Compiling the model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(X_data, Y_data, n_folds=5):
    scores, histories = list(), list()
    # Using five-fold cross validation here
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for i_train, i_test in kfold.split(X_data):
        # Defining model
        model = define_model()
        # Selecting training and testing data
        X_train, Y_train, X_test, Y_test = X_data[i_train], Y_data[i_train], X_data[i_test], Y_data[i_test]
        # Fitting model
        history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test), verbose=0)
        # Evaluate model
        temp, accuracy = model.evaluate(X_test, Y_test, verbose=0)
        print(f"Accuracy > {accuracy*100:.3f}%")
        # Storing scores and histories
        scores.append(accuracy)
        histories.append(history)
    
    return scores, histories

def show_diagnostics(histories):
    for i in range(len(histories)):
        # Loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='training')
        plt.plot(histories[i].history['val_loss'], color='orange', label='testing')
        # Accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='training')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='testing')
    plt.savefig("statistics/diagnostics.png", dpi=300, bbox_inches='tight')
    plt.close()

def summarize_performance(scores):
    print(f"Accuracy: mean={np.mean(scores)*100:.3f}, std={np.std(scores)*100:.3f}, n={len(scores)}")
    plt.boxplot(scores)
    plt.savefig("statistics/summary.png", dpi=300, bbox_inches='tight')
    plt.close()

def test_model():
    X_train, Y_train, X_test, Y_test = load_dataset()
    X_train, X_test = preprocess_pixels(X_train, X_test)
    scores, histories = evaluate_model(X_train, Y_train)
    show_diagnostics(histories)
    summarize_performance(scores)

def run_model():
    X_train, Y_train, X_test, Y_test = load_dataset()
    X_train, X_test = preprocess_pixels(X_train, X_test)
    model = define_model()
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
    model.save("model.keras")

if __name__ == "__main__":
    test_model()
    run_model()