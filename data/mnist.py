from keras.datasets import mnist
import matplotlib.pyplot as plt

def get_mnist():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    for i in range(9):  
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
        plt.show()