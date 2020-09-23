import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков
from functions import train_epohs
from class_neural import neuralNetwork

epoh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
graphs_for_epohs = []
for lr in learning_rate:
    #print(f"{lr}:")
    net = neuralNetwork(784, [100], 10, lr)
    per = []
    for ep in epoh:
        #print(f"\t{ep}:", end=" ")
        for i in range(ep):
            #print(f"{i}", end=" ")
            if i == ep - 1:
                per.append(train_epohs(net, "data_set/mnist_train.csv", "data_set/mnist_test.csv"))
            else:
                train_epohs(net, "data_set/mnist_train.csv", "data_set/mnist_test.csv")
        #print()
    #print()
    graphs_for_epohs.append(per)
colors = [plt.cm.tab10(i/float(len(learning_rate)-1)) for i in range(len(learning_rate))]
for i in range(len(learning_rate)):
    plt.plot(epoh, graphs_for_epohs[i], 'o-r', label=learning_rate[i], color=colors[i])
plt.title("Зависимость эффективности от колличества эпох")
plt.xlabel('Колличество эпох')
plt.ylabel('%')
plt.grid()
plt.legend()
plt.show()
