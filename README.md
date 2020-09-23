# Классическая нейронная сеть по распознаванию цифр с картинки
В данном проекте я реализовал свою первую нерйонную сеть предназначенную для распознавания цифр с картинкок по книге Тарика Рашида ["Создаем нейронную сеть"](https://www.ozon.ru/context/detail/id/141796497/). Основной алгоритм построения взят из книги, но в процессе алгоримт был доработан, и было реализована возможность задавать не только колличество скрытых узлов, но и колличество скрытых слоев. Данный проект реализован в процессе подготовки к дипломной работы, для ознакомления с принципами работы нейронных сетей. Так же статистическим методом была выявлена оптимальная настройка неронной сети для получения наибольшего процента правильных ответов.
## Скачивание проекта и установка необходимых компонентов
Скачайте данный проект архивом или через git, как Вам удобно. Установите Python c [официального сатйа](https://www.python.org/). После установки питона, необходимо скачать библиотеки, задействованные в проекте
```
>pip install numpy
>pip install matplotlib
```
## Запуск программы для демонстрации
Для получения визуальной наглядности используйте какое-либо IDLE(например [PyChar](https://www.jetbrains.com/ru-ru/pycharm/)). Запустите программу ```presentatiom.py```.
![precentation_mode](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/presentation_mode.png)
## Программы для получения различной статистики
## Зависимость эффективности нейронной сети от колличества тренировочных данных
```
>python statistic_db.py
```
![statistic_db](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistic_db.png)

## Зависимость эффективности нейронной сети от колличества скрытых узлов (один скрытый слой)
```
>python statistic_hn.py
```
![statistic_hn_1000-10](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistic_hn_1000-10.png)
## Зависимость эффективности нейронной сети от коэффициента обучаемости
```
>python statistic_lr.py
```
![statistic_lr](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistics_lr.png)
## Зависимость эффективности нейронной сети от колличества эпох обучения
```
>python statistic_epoh.py
```
![statistic_epoh](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistics_epohs.png)
## Зависимость эффективности нейронной сети от колличества эпох обучения для различных коэффициентов обучения
```
>python statistic_epoh_lr.py
```
![statistic_epoh_lr](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistics_opohs_lr.png)
## Зависимость эффективности нейронной сети от колличества скрытых слоев
Для получения данного графика создавались нейронный сети со следующими конфигурациями скрытх слоев:
* 0 скрытых слоев;
* 1 скрытый слой с 500 узлами;
* 2 скрытых слоя по 16 узлов;
* 3 скрытых слоя: 1 слой 500 узлов, 2 - 100, 3 - 16;
```
>python statistic_hn_len.py
```
![statistic_hn_len](https://github.com/Bazarovinc/first_neaural_network/blob/master/imgies/statistics_hn_len.png)
