from functions import train_and_query
import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков

hn = [1000, 500, 100, 16, 10]
per = []
for i in hn:
    print(i)
    per.append(train_and_query(0.2, "data_set/mnist_train.csv", "data_set/mnist_test.csv", i))
plt.plot(hn, per, 'o-r')
plt.title("Зависимость эффективности от колличества скрытых узлов")
plt.xlabel('Колличество скрытых узлов')
plt.ylabel('%')
plt.grid()
plt.show()
