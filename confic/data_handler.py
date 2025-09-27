import pandas as pd
import logging


""" Функции работающие с данными """

logger = logging.getLogger(__name__)


def load_data(file_path):
    """Загрузка данных из CSV файла."""
    try:
        df = pd.read_csv(file_path)
        # Приводим все строки в столбцах к нижнему регистру
        df = df.apply(lambda col: col.str.lower()
                      if col.dtype == "object" else col)
        return df

    except FileNotFoundError:
        logger.error(f"Файл не найден: {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.error(f"Файл пуст: {file_path}")
        return pd.DataFrame()
    except pd.errors.ParserError:
        logger.error(f"Ошибка парсинга файла: {file_path}")
        return pd.DataFrame()


def get_term_definition(term):
    """Получение определения термина и его категории."""
    terms_df = load_data('data/terms.csv')
    result = terms_df.loc[terms_df['term'] == term, ['definition', 'category']]

    if not result.empty:
        definition = result['definition'].iloc[0]
        category = result['category'].iloc[0]
        return definition, category
    else:
        return "Термин не найден.", None


def get_terms_by_category(category):
    """
    Получение списка терминов в указанной категории, 
    отсортированных в алфавитном порядке.
    """
    terms_df = load_data('data/terms.csv')
    terms_in_category = terms_df.loc[terms_df['category'] == category, 'term']
    return sorted(terms_in_category.tolist()) if not terms_in_category.empty else []


def get_strategy_examples(strategy_name):
    """Получение примеров стратегии."""
    strategies_df = load_data('data/strategies.csv')
    examples = strategies_df.loc[strategies_df['strategy_name']
                                 == strategy_name, 'examples']
    return examples.iloc[0] if not examples.empty else "Стратегия не найдена."


def get_strategy_description(strategy_name):
    """Получение описания стратегий"""
    strategies_df = load_data('data/strategies.csv')
    description = strategies_df.loc[strategies_df['strategy_name']
                                    == strategy_name, 'description']
    return description.iloc[0] if not description.empty else "Стратегия не найдена."


def get_content_tips():
    """Получение советов по контенту."""
    content_df = load_data('data/content_tips.csv')
    tips = content_df.sample(3)  # Получаем 3 случайных совета
    return tips.to_string(index=False)


def get_case_studies(case_title):
    """Получение примеров кейсов."""
    cases_df = load_data('data/cases.csv')
    case_study = cases_df.loc[cases_df['case_title'] == case_title, [
        'description', 'results', 'lessons_learned']]
    if not case_study.empty:
        return case_study.to_string(index=False)
    return "Кейс не найден."
