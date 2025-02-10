import numpy as np 
import pandas as pd 
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

path = "/kaggle/input/formula-1-race-data-sqlite/"
database = 'C:\\1KOYARA\\Nebitno\Анализ больших данных\\Lab1\\Formula 1 Race Data\\formula1.sqlite'

# Establish a connection to the SQLite database
try:
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    df = sns.load_dataset('mpg')
    


except Exception as e:
    print("An error occurred:", e)


finally:
    # Close the database connection
    if connection:
        connection.close()