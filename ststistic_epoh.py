import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков
from functions import train_epohs
from class_neural import neuralNetwork

epoh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
percent_for_epohs = []
for e in epoh:
    net = neuralNetwork(784, 100, 10, 0.2)
    print(f"{e}:", end=" ")
    for i in range(e):
        print(f"{i}", end=' ')
        if i == e - 1:
            percent_for_epohs.append(train_epohs(net, "data_set/mnist_train.csv", "data_set/mnist_test.csv"))
        else:
            train_epohs(net, "data_set/mnist_train.csv", "data_set/mnist_test.csv")
    print()
plt.plot(epoh, percent_for_epohs, 'o-r')
plt.title("Зависимость эффективности от колличества эпох")
plt.xlabel('Колличество эпох')
plt.ylabel('%')
plt.grid()
plt.show()
