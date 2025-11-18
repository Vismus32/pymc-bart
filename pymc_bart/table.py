# table.py

import numpy as np
import pandas as pd

class Table:
    def __init__(self, n_rows=5):
        # создаем случайную таблицу с датами и предикатами
        dates = pd.date_range("2025-01-01", periods=n_rows)
        preds = np.random.rand(n_rows)
        self.data = pd.DataFrame({"date": dates, "pred": preds})

    def show(self):
        # функция для вывода таблицы
        print("Таблица из Table:")
        print(self.data)

    def get_pred(self):
        # возвращает массив предикатов для использования в PyMC
        return self.data["pred"].values
