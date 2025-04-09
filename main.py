import os
import re
import logging
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer

from confic.data_handler import *
from confic.semantic_search import *
from confic.response_generator import *

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = 'Model_results/fine_tuned_model/content/model_path/fine_tuned_model'

class MarketingAssistant:
    """Инициализация ассистента с загрузкой модели и токенизатора."""
    def __init__(self, model_name = model_name):
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise
        self.context = []
        self.feedback_data = []
        self.simple_responses = self.load_simple_responses()
        self.load_data_from_self()

    """Загрузка простых ответов из CSV файла и возвращение их в виде словаря."""
    def load_simple_responses(self):
        try:
            responses_df = pd.read_csv('data/simple_responses.csv')
            logging.info("Простые ответы загружены из файла.")
            return dict(zip(responses_df['input'], responses_df['response']))
        except Exception as e:
            logging.error(f"Ошибка при загрузке простых ответов: {e}")
            return {}

    """Функции для сохранения обратной связи"""
    def request_feedback(self, user_input, response):
        feedback = input(f"Вы удовлетворены ответом на вопрос '{user_input}'? (да/нет): ")
        self.feedback_data.append({
            'question': user_input,
            'response': response,
            'feedback': feedback
        })
        logging.info(f"Получена обратная связь: {feedback} на вопрос '{user_input}'.")
        if feedback.lower() == 'нет':
            correction = input("Пожалуйста, укажите, что было не так: ")
            self.learn_from_feedback(user_input, correction)

    """Обработка обратной связи пользователя и сохранение ее в файл."""
    def learn_from_feedback(self, user_input, correction):
        feedback_entry = {
            'question': user_input,
            'correction': correction
        }
        # Сохраняем обратную связь в файл
        save_feedback(feedback_entry)
        logging.info(f"Обратная связь сохранена для вопроса '{user_input}'.")        

    """Загрузка данных из CSV файлов для терминов, стратегий и кейсов."""
    def load_data_from_self(self):
        # Загрузка данных из файлов
        try:
            self.terms = load_data('data/terms.csv')
            self.strategies = load_data('data/strategies.csv')
            self.cases = load_data('data/cases.csv')
            logging.info("Данные загружены из файлов.")
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных: {e}")
    
    """Проверка безопасности"""
    def is_safe_query(self, query):
        unsafe_keywords = ['оскорбление', 'дискриминация', 'порнография']
        result = not any(keyword in query for keyword in unsafe_keywords)
        logging.info(f"Запрос '{query}' безопасен: {result}.")
        return result 
    
    """Приветствие"""
    def greet_user(self):
        return "Привет! Я нейро-маркетолог Полина, ваш проводник в мире маркетинга. Чем могу помочь?"
    
    """Обработка пользовательского ввода, включая проверку на простые вопросы и безопасность."""
    def process_query(self, user_input):
        # Очистка и нормализация входных данных
        user_input = user_input.strip().lower()
        logging.info(f"Обработка запроса пользователя: '{user_input}'.")
        
        # Проверка на простые вопросы
        if user_input in self.simple_responses:
            logging.info("Запрос найден в простых ответах.")
            return self.simple_responses[user_input]
    
        if not self.is_safe_query(user_input):
            return "Ваш запрос содержит небезопасные слова."
        
        # Извлечение релевантной информации
        retrieved_info = self.retrieve_relevant_info(user_input)
        # Объединение контекста с извлеченной информацией
        context_str = "\n".join(self.context)
        
        if retrieved_info:
            context_str +=  f"\nИзвлеченная информация: {retrieved_info}\nПользователь: {user_input}\nПолина: "
        else:
            context_str += f"\nПользователь: {user_input}\nПолина: "
            
        # Ограничиваем длину контекста
        input_ids = self.tokenizer.encode(context_str, return_tensors='pt')

        # Обрезаем до максимальной длины
        max_length = self.model.config.n_positions  
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]

        response = self.generate_response(user_input)
        logging.info(f"Сгенерирован ответ: '{response}' для запроса: '{user_input}'.")
        
        self.request_feedback(user_input, response)
        return response
    
    
    """Функция для извлечения релевантной информации из данных."""
    def retrieve_relevant_info(self, user_input):
        # Пример извлечения на основе ключевых слов
        key_words = user_input.split()  # Разбиваем ввод на ключевые слова
        relevant_info = []

        # Проверка в каждой категории данных
        for term in self.terms['term']:
            if any(keyword in term.lower() for keyword in key_words):
                relevant_info.append(term)
                
        return ", ".join(relevant_info) if relevant_info else None


    """Определение эмоциональной окраски"""
    def detect_intent(self, user_input):
        emotional_state = self.analyze_emotion(user_input)
        
        if emotional_state == 'негативный':
            return self.empathic_response(user_input)
        
        elif any(phrase in user_input for phrase in ["что такое", "определение", "термин"]):
            return self.handle_definition(user_input)
    
        elif "примеры стратегии" in user_input:
            return self.handle_strategy_examples(user_input)
        elif "советы по контенту" in user_input:
            return self.get_content_tips()
        elif "кейсы" in user_input:
            return self.handle_case_studies(user_input)
        else:
            return self.generate_response(user_input)
        
        
    """Простейший анализ тональности на основе ключевых слов"""    
    def analyze_emotion(self, user_input):
        negative_words = ['плохо', 'трудно', 'беспокойство', 'неуверенность']
        positive_words = ['хорошо', 'удачно', 'успех', 'радость']
        
        if any(word in user_input for word in negative_words):
            return 'негативный'
        elif any(word in user_input for word in positive_words):
            return 'позитивный'
        return 'нейтральный'    

    def empathic_response(self, user_input):
        return "Мне так жаль что возникают трудности. Могу предложить несколько советов или ресурсов, которые могут помочь."
    
    
    """Функия быстрого нахождения термина"""
    def handle_definition(self,user_input):
        term = re.sub(r'^(определение|что такое|термин)\s*', '', user_input.lower()).strip()
        
        definition, category = get_term_definition(term)
        
        if definition != "Термин не найден.":
            print(f"{term}: {definition}")
            follow_up = input("Хотите узнать практическое применение термина? (да/нет) ").strip().lower()
            
            if follow_up == "да":
                    examples = get_strategy_examples(term)
                    print(f"Примеры использования {term}: {examples}")
                    follow_up_term = input(f"{term} относится к категории {category}, хотите получить перечень терминов попадающих в эту категорию? (Да/Нет)").strip().lower()
                    
                    if follow_up_term == "да":
                        terms_list = get_terms_by_category(category)
                        if terms_list:
                            print("Список терминов в категории:")
                            for t in terms_list:
                                print(f"- {t}")
                       
        else:
            results = (semantic_search(term, self.terms['term'].tolist()))[:5]
            return "Нет данных. Похожие термины: " + ", ".join([similar_term for similar_term, _ in results])

        
    """Функция поиска стратегий"""
    def handle_strategy_examples(self, user_input):
        strategy_name = user_input.replace("примеры стратегии ", "").strip()
        examples = self.get_strategy_examples(strategy_name)
        if examples != "Стратегия не найдена.":
            return f"Примеры для {strategy_name}: {examples}"
        else:
            results = self.semantic_search(strategy_name, self.strategies)
            if results:
                return "Стратегия не найдена. Похожие стратегии: " + ", ".join([similar_strategy for similar_strategy, _ in results])
            return "Стратегия не найдена."
        
    
    """Функция поиска кейсов"""
    def handle_case_studies(self, user_input):
        case_title = user_input.replace("кейсы ", "").strip()
        case_study = self.get_case_studies(case_title)
        if case_study != "Кейс не найден.":
            return f"Кейс {case_title}: {case_study}"
        else:
            results = self.semantic_search(case_title, self.cases)
            if results:
                return "Кейс не найден. Похожие кейсы: " + ", ".join([similar_case for similar_case, _ in results])
            return "Кейс не найден."

    
    """Функция для генерации ответа с использованием модели машинного обучения"""
    def generate_response(self, user_input):
        # Проверяем, что user_input является тензором и извлекаем данные из тензора
        if isinstance(user_input, torch.Tensor):
            user_input = user_input.squeeze().tolist()  
        
        # Если user_input - это список, создаем строку
        if isinstance(user_input, list):
            user_input = " ".join(map(str, user_input))
        
        # Проверяем, есть ли действительный ввод
        if not user_input.strip():
            return "Пожалуйста, задайте вопрос."
        
        context_str = "\n".join(self.context) + f"\nПользователь: {user_input}\nПолина: "
        input_ids = self.tokenizer.encode(context_str, return_tensors='pt')

        # Устанавливаем pad_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Создаем attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        response = self.model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            do_sample=True, 
            temperature=0.5,
            top_k=100,
            top_p=0.85,
            attention_mask=attention_mask  
        )
        
        decoded_response = self.tokenizer.decode(response[0], skip_special_tokens=True)
        response_text = decoded_response.split("Полина:")[-1].strip() if "Полина:" in decoded_response else decoded_response.strip()
        response_text = response_text.replace("…", "").strip()
        
        if not response_text or len(response_text) < 5:
            return "Извини, я не совсем поняла. Можешь переформулировать вопрос?"

        return response_text


    """Основная функция для взаимодействия с пользователем."""   
    def run(self):
        print(self.greet_user())
        while True:
            user_input = input("Введите свой запрос (или 'exit' для выхода): ")
            
            if user_input.lower() == 'exit':
                print("Спасибо за общение! Если будут вопросы, возвращайтесь!")
                break
            
            if not user_input.strip():
                print("Пожалуйста, введите что-то.")
                continue
            
            print(self.process_query(user_input))
            
      
    """Функция для получения данных о посещаемости"""
    def get_google_analytics_data(self, view_id):
        key='YOUR_API_KEY'
        url = f"https://analytics.googleapis.com/v3/data/ga?ids=ga:{view_id}&metrics=ga:sessions,ga:users&key=YOUR_API_KEY"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return "Не удалось получить данные из Google Analytics."

    """Функция с данными о производительности, которые могут быть использованы для анализа и мониторинга."""
    def analyze_performance(self):
        view_id = 'YOUR_VIEW_ID'
        data = self.get_google_analytics_data(view_id)
        return f"Данные о производительности: {data}"
       

if __name__ == "__main__":
    assistant = MarketingAssistant()
    assistant.run()
    
    
    

