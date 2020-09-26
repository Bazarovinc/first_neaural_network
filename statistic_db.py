import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков
from functions import train_and_query  # импортируем функцию для проучения проценат правильных ответов

percent = []
# получение процента правильных ответов для разных баз данных
percent.append(train_and_query(0.3, "data_set/mnist_train.csv", "data_set/mnist_test.csv", [100]))
percent.append(train_and_query(0.3, "data_set/mnist_train_100.csv", "data_set/mnist_test_10.csv", [100]))
amount = ["60000", "100"]
fig, ax = plt.subplots()
# построение инфографика для разных бд
ax.bar(amount, percent, width=0.5)
# вклюение сетки
plt.grid()
# заголовок графика
plt.title('Зависимость эффективности от колличества тренировочных данных')
# название оси x
plt.xlabel('Колличество тренировочных данных')
# название оси y
plt.ylabel('%')
# вывод графика
plt.show()
