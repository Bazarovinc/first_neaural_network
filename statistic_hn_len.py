from functions import train_and_query
import matplotlib.pyplot as plt  # импортируем библиотеку для построения графиков

hn = [[], [500], [16, 16], [500, 100, 16]]
len_hn = [len(i) for i in hn]
per = []
for i in hn:
    print(i)
    per.append(train_and_query(0.2, "data_set/mnist_train.csv", "data_set/mnist_test.csv", i))
plt.plot(len_hn, per, 'o-r')
print(per)
plt.title("Зависимость эффективности от колличества скрытых слоёв")
plt.xlabel('Колличество скрытых слоёв')
plt.ylabel('%')
plt.grid()
plt.show()
