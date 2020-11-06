# Классическая нейронная сеть по распознаванию цифр с изображения
В данном проекте я реализовал свою первую нерйонную сеть, предназначенную для распознавания цифр с картинкок по книге Тарика Рашида ["Создаем нейронную сеть"](https://www.ozon.ru/context/detail/id/141796497/). Основной алгоритм построения взят из книги, но в процессе был доработан, и было реализована возможность задавать не только колличество скрытых узлов, как в книге, но и колличество скрытых слоев. Данный проект реализован в процессе подготовки к дипломной работые, для ознакомления с принципами работы нейронных сетей. Так же статистическим методом была выявлена оптимальная настройка неронной сети для получения наибольшего процента правильных ответов.
# Что такое нейронная сеть?
Попытаюсь кратко изложить суть неронной сети в данной задачи, и необходимость использования нейронной сети в задачи распознавания рукописных чисел. Нейронная сеть - это  математическая модель, а также её программное воплощение, построенная по принципу организации и функционирования биологических нейронных сетей.
![neural_network](https://wiki.loginom.ru/images/multilayer-neural-net.svg)

Каждый нейрон может принимать, подобно биологическому нейрону, входной и выходной сигналы. Объединяя большое колличество нейронов мы получаем нейронную сеть.
Любая нейронная сеть имеет входной слой нейронов, куда принимает данные, скрытый слой(и), где идет обработка данных(или может не иметь скрытых слоев), и выходной слой, где мы получаем какой-либот ответ/результат.
### Определим некоторые понятия
* Входной слой

Во входном слое все нейроны принимают лишь один сигнал и имеют множество связей со скрытым(выходным) слоем, куда на каждый нейрон последующего слоя отдается выходной сигнал.
* Скрытый слой(слои)

В скрытом слое(ях) каждый нейрон принимает сигнал от каждого нейрона входного слоя, пропускает его через функцию активации(сглаживания(про нее позднее)) и отдает на каждый нейрон следующего слоя.
* Выходной слой.

В выходном слое каждый нейрон принимает сигнал от каждого нейрона скрытого слоя, пропускает его через функцию активации(сглаживания) и мы получаем результат работы нейронной сети.
* Связи нейронов

Каждый нейрон имеет связь, которая имеет коэффициент или весы. Именно улчешение значения весов или уравновешивание и является основной идеей обучения нейронной сети.
* Функция сглаживания

Функция сглаживания или функция активации взята по примеру реальных биологических нейронов, и в ее основе лежит математическая функция сигмоиды(в данной модели нейронной сети). Идея использования функции активации заключается в том, чтобы сгладить входной сигнал нейрона.
* Входной сигнал

Входной сигнал - это сумма всех сигналов полученных от предыдущего слоя нейронов и умноженных на свой весовой коэффициент, поступающих на каждый отдельный нейрон (кроме входного сигнала).
* Выходной сигнал

Выходной сигнал - это сигнал, пропущенный через функцию активации, и отданный далее: либо на следующий слой неронов, либо на выход.
### Формирование весовых коэффициентов
Все весовые коэффициенты формируются при создании нейронной сети. Формирование их происходит случайным образом. Значения коэффициентов принимают значения от -0.99 до 0.99, исключая 0.
### Определение колличества нейронов в каждом слое
Колличество нейрнов во входном и выходном слоях определяется задачей. Так в задаче по распознаванию рукописных чисел колличество нейрнов во входном слое равно 784 по причине то, что изображение с числом представляет собой 28 на 28 пикселей. То есть значение каждого пикселя является входным данным. Колличество нейронов в выходном слое равно 10, т.к. нейронная сеть распознает число от 0 до 9. И максимальный выходной сигнал будет определять то, какую цифру определила нейронная сеть. Колличество нейронов в скрытом слое определяется статистическим подбором(и не должно быть больше предыдущего слоя). Так я подобрал, что в данной задаче эффективнее иметь один скрытый слой состоящий из 500 нейронов.
### Процесс работы нейронной сети
Суть выдачи ответа нейронной сети является лишь набором математических операций. Как вы уже поняли, входной и выходной сегналы являются векторами равными колличеству нейронов. 
## Скачивание проекта и установка необходимых компонентов
Скачайте данный проект архивом или через git, как Вам удобно. Установите Python c [официального сатйа](https://www.python.org/). После установки питона, необходимо скачать библиотеки, задействованные в проекте
```
>pip install numpy
>pip install matplotlib
```
## Запуск программы для демонстрации
Для получения визуальной наглядности используйте какое-либо IDLE(например [PyCharm](https://www.jetbrains.com/ru-ru/pycharm/)). Запустите программу ```presentation.py```. После запуска Вы будете получать картинку с цифрой, результаты выдаваемые нейронной сетью и цифру, которую должны были получить.
![precentation_mode](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/presentation_mode.png)
## Программы для получения различной статистики
### Зависимость эффективности нейронной сети от колличества тренировочных данных
```
>python statistic_db.py
```
![statistic_db](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistic_db.png)

### Зависимость эффективности нейронной сети от колличества скрытых узлов (один скрытый слой)
```
>python statistic_hn.py
```
![statistic_hn_1000-10](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistic_hn_1000-10.png)
### Зависимость эффективности нейронной сети от коэффициента обучаемости
```
>python statistic_lr.py
```
![statistic_lr](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistics_lr.png)
### Зависимость эффективности нейронной сети от колличества эпох обучения
```
>python statistic_epoh.py
```
![statistic_epoh](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistics_epohs.png)
### Зависимость эффективности нейронной сети от колличества эпох обучения для различных коэффициентов обучения
```
>python statistic_epoh_lr.py
```
![statistic_epoh_lr](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistics_opohs_lr.png)
### Зависимость эффективности нейронной сети от колличества скрытых слоев
Для получения данного графика создавались нейронный сети со следующими конфигурациями скрытх слоев:
* 0 скрытых слоев;
* 1 скрытый слой с 500 узлами;
* 2 скрытых слоя по 16 узлов;
* 3 скрытых слоя: 1 слой 500 узлов, 2 - 100, 3 - 16.
```
>python statistic_hn_len.py
```
![statistic_hn_len](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistic_hn_len.png)
## Результаты проведенного статистического анализа
Получив статистику, изменяя различные конфигурации нейронной сети, можно сформировать два варианта идеальных конфигураций сети:
* Первый
```
Коэффициент обучаемости - 0.2
Один скрытый слой с 500 узлами
Одна эпоха обучения
Обучение на полноценной БД MNIST, состоящей из 60000 тренировчных данных
```
* Второй
```
Коэффициент обучаемости - 0.1
Один скрытый слой с 500 узлами
Три эпохи обучения
Обучение на полноценной БД MNIST, состоящей из 60000 тренировчных данных
```
