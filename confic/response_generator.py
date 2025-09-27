
import logging
import torch


def generate_response(input_text, model, tokenizer):
    """Генерирует ответ на основе входного текста,
    используя заданную модель и токенизатор
    """
    # Токенизация входного текста
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )

    # Декодирование и возвращение ответа
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Полина:" in response:
        response_text = response.split("Полина:")[-1].strip()
    else:
        response_text = response.strip()

    print(f"Сгенерированный ответ: {response_text}")

    return response_text


def log_interaction(user_input, response):
    """
    Логирует взаимодействие пользователя с нейросотрудником
    """
    logging.info(f"User: {user_input}, Response: {response}")


def is_safe_query(query):
    """Проверка безопасности"""
    unsafe_keywords = ['оскорбление', 'дискриминация', 'порнография']
    return not any(keyword in query for keyword in unsafe_keywords)
