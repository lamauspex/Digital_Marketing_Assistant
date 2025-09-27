
"""Функционал аналитики"""


import requests


def get_google_analytics_data(view_id):
    key = 'YOUR_API_KEY'
    url = (
        f"https://analytics.googleapis.com/v3/data/ga?"
        f"ids=ga:{view_id}&metrics=ga:sessions,ga:users&key={key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return "Не удалось получить данные из Google Analytics."


def analyze_performance(assistant):
    view_id = 'YOUR_VIEW_ID'
    data = assistant.get_google_analytics_data(view_id)
    return f"Данные о производительности: {data}"
