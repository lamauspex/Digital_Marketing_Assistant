
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import json
import pandas as pd

# Загрузка модели для семантического поиска
semantic_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


# Функции для семантического поиска,
# генерации ответов и проверки безопасности
def save_feedback(feedback_data, file_path='feedback.json'):
    """Сохраняет обратную связь в файл."""
    with open(file_path, 'a') as f:
        json.dump(feedback_data, f)
        f.write('\n')


def load_feedback(file_path='feedback.json'):
    """Загружает обратную связь из файла."""
    feedback_list = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                feedback_list.append(json.loads(line))
    except FileNotFoundError:
        logging.error("Файл обратной связи не найден.")
    return feedback_list


def semantic_search(query, data, top_k=3):
    """
    Выполняет семантический поиск,
    находит наиболее похожие элементы на основе входного запроса.

    :param query: Запрос, на основе которого будет произведен поиск.
    :param data: Список данных, среди которых будет производиться поиск.
    :param top_k: Количество лучших результатов, которые нужно вернуть.
    :return: Список кортежей, содержащих элементы данных и их соответствующие оценки схожести.
    """

    # Убедитесь, что data - это список строк
    if isinstance(data, pd.DataFrame):
        # Если data - DataFrame, используем только нужный столбец
        data = data['term'].tolist()

    # Получение векторов для запроса и данных
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    data_embeddings = semantic_model.encode(data, convert_to_tensor=True)

    # Вычисление косинусного сходства
    cos_sim = util.pytorch_cos_sim(query_embedding, data_embeddings)

    # Получение индексов лучших результатов
    top_results = np.argpartition(-cos_sim.cpu().detach().numpy(),
                                  range(top_k))[:top_k]
    top_results = top_results.flatten()
    return [(data[i], cos_sim[0][i].item()) for i in top_results]
