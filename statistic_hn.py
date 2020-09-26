from functions import train_and_query  # импортируем функцию для проучения проценат правильных ответов
import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков

hn = [[1000], [500], [100], [16], [10]]  # задаем массив скрытых узлов
per = []
# получение процента правильных ответов в зависимости от колличества скрытых узлов
for i in hn:
    print(i)
    per.append(train_and_query(0.2, "data_set/mnist_train.csv", "data_set/mnist_test.csv", i))
plt.plot(hn, per, 'o-r')  # построение графика
plt.title("Зависимость эффективности от колличества скрытых узлов")  # заголовок
plt.xlabel('Колличество скрытых узлов')  # ось x
plt.ylabel('%')  # ось y
plt.grid()  # сетка
plt.show()  # вывод графика
