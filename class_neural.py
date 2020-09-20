import numpy as np  # импортирование библиотеки для работы с матрицами, векторами
import scipy.special as sc  # импортирование библиотеки для получения функции сигмоиды


# определение класса нейронной сети
class neuralNetwork:
    # инициализирование нейронной сети
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # задаем количество узлов во входном, скрытом и выходном слое
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        # коэффициент обучения
        self.lr = learning_rate
        # Матрицы весовых коэффициентов связей wih (между входным и скрытым слоями)
        # и who (между скрытым и выходным слоями).
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: sc.expit(x)

    # метод для тренировки нейронной сети
    def train(self, inputs_list, targets_list):
        # преобразовать список входных значений в двухмерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входных значений в двухмерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
