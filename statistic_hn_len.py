from functions import train_and_query  # импортируем функцию для проучения проценат правильных ответов
import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков

hn = [[], [500], [16, 16], [500, 100, 16]]  # задаем список со скрытым слоями и узлами в них
len_hn = [len(i) for i in hn]  # полчуние списка с колличеством скрытых слоев для простроения граифка
per = []
# получение процента правильных ответов для каждого элемента из списка hn
for i in hn:
    #print(i)
    per.append(train_and_query(0.2, "data_set/mnist_train.csv", "data_set/mnist_test.csv", i))
plt.plot(len_hn, per, 'o-r')  # построение графика
#print(per)
plt.title("Зависимость эффективности от колличества скрытых слоёв")  # заголовок
plt.xlabel('Колличество скрытых слоёв')  # ось x
plt.ylabel('%')  # ось y
plt.grid()  # сетка
plt.show()  # вывод графика
