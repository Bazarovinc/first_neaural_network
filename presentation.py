from class_neural import neuralNetwork  # импортирование класса нейронной сети
import numpy as np  # импортируем библиотек для работы с массивами
import matplotlib.pyplot  # импортируем библиотеку для построения графиков
# количество входных, скрытых и выходных узлов
# задано 784 узла на входном слое т.к. размер картинок с цирфами 23 пикселя на 23 пикселя
input_nodes = 784
# переход от большого числа узлов к меньшему
hidden_nodes = [500]
# задано 10 узлов на выходе, чтобы в итоге получать число от 0 до 9
output_nodes = 10
# коэффициент обучения равен 0,2
learning_rate = 0.2
# создание экземпляр нейронной сети
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# задание именни файла с тренировочными данными
file_name_train = "data_set/mnist_train.csv"
# конструкция для чтения данных из файла по линиям и закрыти файла
with open(file_name_train, 'r') as f_o:
    data_list = f_o.readlines()
# перемешевиние дата сета с числами
np.random.shuffle(data_list)
for elem in data_list:
    # разаделение считанной строки запятыми
    all_values = elem.split(',')
    # asfarray() преобразует тип входного массива к вещественному типу float64. так же идет форматирование входных
    # данных для подачи их в нейронную сеть
    scaled_input = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    # создание корректных выходных данных. np.zeros создает массив 0 размером output_nodes. все занчения становятся 0.01
    targets = np.zeros(output_nodes) + 0.01
    # задание значения 0.99 у того элемента, цира которого на картинке
    targets[int(all_values[0])] = 0.99
    # подача входных и выходных данных для тренировки нейронки
    n.train(scaled_input, targets)
# задание именни файла с тестовым данными
file_name_test = "data_set/mnist_test.csv"
# конструкция для чтения данных из файла по линиям и закрыти файла
with open(file_name_test, 'r') as f_o:
    test_list = f_o.readlines()
# перемешевиние дата сета с числами, чтобы не смотреть на одни и те же каждый раз
np.random.shuffle(test_list)
for elem in test_list:
    # все тоже, что и в предыдущем цикле, что был для тренировки сети
    all_values = elem.split(',')
    scaled_input = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    # запись корректной цифры
    correct_output = all_values[0]
    # получение выходных значений при подачи тесового числа + транспонирование
    result = np.transpose(n.query(scaled_input))
    # получение индекса максимального выходного сигнала
    output_num = np.argmax(result)
    # попиксельный вывод цифры поданной на тест. reshape((28, 28)) - формирует матрицу 28 на 28 из массива 784 элементов
    matplotlib.pyplot.imshow(scaled_input.reshape((28, 28)), cmap='Greys')
    # вывод графика
    matplotlib.pyplot.show()
    # вывод индекса макимального выходного сигнала из массива для анализа полученного результата и вывод результата,
    # заложенного в дата сет для сравнения
    print(f"Полученно: {output_num} - должно: {all_values[0]}")
    # ожиданние ввода для того, чтобы продолжить цикл
    input()
