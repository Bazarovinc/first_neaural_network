import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков
from functions import train_and_query  # импортируем функцию для получения проценат правильных ответов


learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # задаем список коэффициентов обучения
percent = []
# получаем процент правильных ответов для кажого коэффициента обучения
for lr in learning_rate:
    print(lr)
    percent.append(train_and_query(lr, "data_set/mnist_train.csv", "data_set/mnist_test.csv", [100]))
plt.plot(learning_rate, percent, 'o-r')  # построение графика
plt.title("Зависимость эффективности от коэффициента обучения")  # заголовок
plt.xlabel('Коэффициент обучения')  # ось x
plt.ylabel('%')  # ось y
plt.grid()  # сетка
plt.show()  # вывод графика
