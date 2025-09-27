
""" Работа с датасетом """


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TextDataset, GPT2Tokenizer


def prepare_data(csv_path):
    """Подготовка данных из CSV для дальнейшего разделения."""
    data = pd.read_csv(csv_path)
    train_data, val_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )
    return train_data, val_data


def create_text_datasets(train_data, val_data, block_size=128):
    """Создание тренировочного и валидационного набора данных."""
    train_dataset = TextDataset(
        # Используем стандартный токенизатор gpt2
        tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
        file_path=train_data,
        block_size=block_size
    )
    val_dataset = TextDataset(
        tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
        file_path=val_data,
        block_size=block_size
    )
    return train_dataset, val_dataset
