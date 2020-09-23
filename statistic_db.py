import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков
from functions import train_and_query

percent = []
percent.append(train_and_query(0.3, "data_set/mnist_train.csv", "data_set/mnist_test.csv", [100]))
percent.append(train_and_query(0.3, "data_set/mnist_train_100.csv", "data_set/mnist_test_10.csv", [100]))
amount = ["60000", "100"]
fig, ax = plt.subplots()
ax.bar(amount, percent, width=0.5)
plt.grid()
plt.title('Зависимость эффективности от колличества тренировочных данных')
plt.xlabel('Колличество тренировочных данных')
plt.ylabel('%')
plt.show()
