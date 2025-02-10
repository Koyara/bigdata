import sqlite3

# Подключение к базе данных
conn = sqlite3.connect('C:\\1KOYARA\\Nebitno\Анализ больших данных\\Lab1\\Formula 1 Race Data\\formula1.sqlite')  # Укажите путь к вашему файлу базы данных
cursor = conn.cursor()

# Пример запроса для проверки подключения
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Таблицы в базе данных:", tables)

# Закрытие соединения
conn.close()

import pandas as pd
import matplotlib.pyplot as plt

# Подключение к базе данных
conn = sqlite3.connect('C:\\1KOYARA\\Nebitno\Анализ больших данных\\Lab1\\Formula 1 Race Data\\formula1.sqlite')
df_results = pd.read_sql_query("SELECT points, grid, position FROM results", conn)

# Построение гистограмм
plt.figure(figsize=(15, 5))

# Гистограмма для points
plt.subplot(1, 3, 1)
plt.hist(df_results['points'], bins=20, color='blue', alpha=0.7)
plt.title('Распределение очков')
plt.xlabel('Очки')
plt.ylabel('Частота')

# Гистограмма для grid
plt.subplot(1, 3, 2)
plt.hist(df_results['grid'], bins=20, color='green', alpha=0.7)
plt.title('Распределение стартовой позиции')
plt.xlabel('Стартовая позиция')
plt.ylabel('Частота')

# Гистограмма для position
plt.subplot(1, 3, 3)

# Преобразование типов
df_results['position'] = pd.to_numeric(df_results['position'], errors='coerce')

# Удаление пропусков
df_results = df_results.dropna(subset=['position'])

plt.hist(df_results['position'], bins=20, color='red', alpha=0.7)
plt.title('Распределение позиции в гонке')
plt.xlabel('Позиция')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()