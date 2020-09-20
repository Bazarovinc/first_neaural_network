from class_neural import neuralNetwork
import numpy as np


def train_and_query(lr, train_file, test_file, hid_not):
    # количество входных, скрытых и выходных узлов
    # задано 784 узла на входном слое т.к. размер картинок с цирфами 23 пикселя на 23 пикселя
    input_nodes = 784
    # переход от большого числа узлов к меньшему
    hidden_nodes = hid_not
    # задано 10 узлов на выходе, чтобы в итоге получать число от 0 до 9
    output_nodes = 10
    # коэффициент обучения задается в интервале от 0.1 до 0.9
    learning_rate = lr
    # создание экземпляр нейронной сети
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # задание именни файла с тренировочными данными
    file_name_train = train_file
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
        # создание корректных выходных данных. np.zeros создает массив 0 размером output_nodes.
        # все занчения становятся 0.01
        targets = np.zeros(output_nodes) + 0.01
        # задание значения 0.99 у того элемента, цира которого на картинке
        targets[int(all_values[0])] = 0.99
        # подача входных и выходных данных для тренировки нейронки
        n.train(scaled_input, targets)
    # задание именни файла с тестовым данными
    file_name_test = test_file
    # конструкция для чтения данных из файла по линиям и закрыти файла
    with open(file_name_test, 'r') as f_o:
        test_list = f_o.readlines()
    # перемешевиние дата сета с числами, чтобы не смотреть на одни и те же каждый раз
    np.random.shuffle(test_list)
    scorecard = []
    for elem in test_list:
        # все тоже, что и в предыдущем цикле, что был для тренировки сети
        all_values = elem.split(',')
        scaled_input = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
        # запись корректной цифры
        correct_output = int(all_values[0])
        # получение выходных значений при подачи тесового числа + транспонирование
        result = n.query(scaled_input)
        # запись выходных значений для удобного нахождения максимального выходного сигнала
        output_num = np.argmax(result)
        if correct_output == output_num:
            scorecard.append(1)
        else:
            scorecard.append(0)
    scorecard_array = np.asarray(scorecard)
    per = (scorecard_array.sum() / scorecard_array.size) * 100
    return per


def train_epohs(net, train_file, test_file):
    file_name_train = train_file
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
        # создание корректных выходных данных. np.zeros создает массив 0 размером output_nodes.
        # все занчения становятся 0.01
        targets = np.zeros(10) + 0.01
        # задание значения 0.99 у того элемента, цира которого на картинке
        targets[int(all_values[0])] = 0.99
        # подача входных и выходных данных для тренировки нейронки
        net.train(scaled_input, targets)
    # задание именни файла с тестовым данными
    file_name_test = test_file
    # конструкция для чтения данных из файла по линиям и закрыти файла
    with open(file_name_test, 'r') as f_o:
        test_list = f_o.readlines()
    # перемешевиние дата сета с числами, чтобы не смотреть на одни и те же каждый раз
    np.random.shuffle(test_list)
    scorecard = []
    for elem in test_list:
        # все тоже, что и в предыдущем цикле, что был для тренировки сети
        all_values = elem.split(',')
        scaled_input = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
        # запись корректной цифры
        correct_output = int(all_values[0])
        # получение выходных значений при подачи тесового числа + транспонирование
        result = net.query(scaled_input)
        # запись выходных значений для удобного нахождения максимального выходного сигнала
        output_num = np.argmax(result)
        if correct_output == output_num:
            scorecard.append(1)
        else:
            scorecard.append(0)
    scorecard_array = np.asarray(scorecard)
    per = (scorecard_array.sum() / scorecard_array.size) * 100
    return per