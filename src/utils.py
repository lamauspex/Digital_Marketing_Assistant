
""" Вспомогательные функции """


import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df.to_dict('records')  # Преобразуем DataFrame в словарь записей


def save_feedback(feedback_entry):
    with open('feedback_log.txt', 'a') as file:
        file.write(str(feedback_entry) + '\n')
