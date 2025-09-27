
""" Логика обработки запросов """


def process_query(user_input, assistant):
    cleaned_input = user_input.strip().lower()

    if cleaned_input in assistant.simple_responses:
        return assistant.simple_responses[cleaned_input]

    if not assistant.is_safe_query(cleaned_input):
        return "Ваш запрос содержит небезопасные слова."

    retrieved_info = assistant.retrieve_relevant_info(cleaned_input)
    context_str = "\n".join(assistant.context)

    if retrieved_info:
        context_str += (
            f"\nИзвлеченная информация: {retrieved_info}\n"
            f"Пользователь: {user_input}\n"
            f"Полина: "
        )
    else:
        context_str += f"\nПользователь: {user_input}\nПолина: "

    response = assistant.generate_response(cleaned_input)
    assistant.request_feedback(cleaned_input, response)
    return response
