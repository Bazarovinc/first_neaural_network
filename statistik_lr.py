import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков
from functions import train_and_query


learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
percent = []
for lr in learning_rate:
    print(lr)
    percent.append(train_and_query(lr, "data_set/mnist_train.csv", "data_set/mnist_test.csv", [100]))
plt.plot(learning_rate, percent, 'o-r')
plt.title("Зависимость эффективности от коэффициента обучения")
plt.xlabel('Коэффициент обучения')
plt.ylabel('%')
plt.grid()
plt.show()
