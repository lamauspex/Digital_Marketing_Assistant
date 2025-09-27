
""" Основной класс ассистента """


import logging
import pandas as pd

from src.handlers import process_query
from .models import load_model_and_tokenizer
from .utils import load_data, save_feedback


class MarketingAssistant:
    def __init__(
        self,
        model_name=(
            "Model_results/fine_tuned_model/"
            "content/model_path/fine_tuned_model"
        )
    ):
        self.model_name = model_name
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        self.context = []
        self.feedback_data = []
        self.simple_responses = self.load_simple_responses()
        self.load_data_from_files()

    def load_simple_responses(self):
        try:
            responses_df = pd.read_csv('data/simple_responses.csv')
            logging.info("Простые ответы загружены из файла.")
            return dict(zip(responses_df['input'], responses_df['response']))
        except Exception as e:
            logging.error(f"Ошибка при загрузке простых ответов: {e}")
            return {}

    def load_data_from_files(self):
        try:
            self.terms = load_data('data/terms.csv')
            self.strategies = load_data('data/strategies.csv')
            self.cases = load_data('data/cases.csv')
            logging.info("Данные загружены из файлов.")
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных: {e}")

    def is_safe_query(self, query):
        unsafe_keywords = ['оскорбление', 'дискриминация', 'порнография']
        return not any(keyword in query for keyword in unsafe_keywords)

    def greet_user(self):
        return ("Привет! Я нейро-маркетолог Полина,"
                "ваш проводник в мире маркетинга."
                "Чем могу помочь?")

    def retrieve_relevant_info(self, user_input):
        key_words = user_input.split()
        relevant_info = [
            term
            for term in self.terms['term']
            if any(keyword in term.lower() for keyword in key_words)
        ]
        return ", ".join(relevant_info) if relevant_info else None

    def generate_response(self, user_input):
        context_str = "\n".join(self.context) + \
            f"\nПользователь: {user_input}\nПолина: "
        input_ids = self.tokenizer.encode(context_str, return_tensors='pt')
        response = self.model.generate(
            input_ids, max_length=150, num_return_sequences=1)[0]
        decoded_response = self.tokenizer.decode(
            response, skip_special_tokens=True)
        return decoded_response.split("Полина:")[-1].strip()

    def request_feedback(self, user_input, response):
        feedback = input(
            f"Вы удовлетворены ответом на вопрос '{user_input}'? (да/нет): ")
        self.feedback_data.append(
            {
                'question': user_input,
                'response': response,
                'feedback': feedback}
        )
        if feedback.lower() == 'нет':
            correction = input("Пожалуйста, укажите, что было не так: ")
            self.learn_from_feedback(user_input, correction)

    def learn_from_feedback(self, user_input, correction):
        feedback_entry = {'question': user_input, 'correction': correction}
        save_feedback(feedback_entry)
        logging.info(f"Обратная связь сохранена для вопроса '{user_input}'.")

    def run(self):
        print(self.greet_user())
        while True:
            user_input = input("Введите свой запрос (или 'exit' для выхода): ")
            if user_input.lower() == 'exit':
                print("Спасибо за общение! Если будут вопросы, возвращайтесь!")
                break
            print(process_query(user_input, self))
