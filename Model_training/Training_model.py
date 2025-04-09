import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



# Загрузка данных из CSV
data = pd.read_csv('data/training_baza.csv')

# Разделение на обучающую и валидационную выборки
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Сохранение данных в текстовые файлы для обучения
train_data['text'].to_csv('train_data.txt', index=False, header=False)
val_data['text'].to_csv('val_data.txt', index=False, header=False)

# Определение пути для сохранения модели
DATA_PATH = Path('Model_results')

class FineTuner:
    def __init__(self,
                 model_name='gpt2',
                 cache_dir='model_cache',
                 data_path=DATA_PATH):
        self.data_path = Path(data_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=str(self.data_path / cache_dir))
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=str(self.data_path / cache_dir))
        self.train_losses = []
        self.val_losses = []  # Добавляем список для потерь валидации

    def fine_tune(self, train_file, val_file, output_name='fine_tuned_model',
                  num_train_epochs=10, per_device_train_batch_size=2,
                  learning_rate=5e-5, save_steps=1000):
        print("Fine tuner object:", self)

        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=str(train_file),
            block_size=128
        )
        val_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=str(val_file),
            block_size=128
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=str(self.data_path / output_name),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=save_steps,
            learning_rate=learning_rate,
            save_total_limit=2,
            logging_dir=str(self.data_path / 'logs'),
            logging_steps=100,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics  # Добавление метрик
        )

        trainer.train()
        
        # Сохранение потерь после обучения
        self.train_losses = trainer.state.log_history

        # Сохранение обученной модели и токенизатора
        self.model.save_pretrained(str(self.data_path / output_name))
        self.tokenizer.save_pretrained(str(self.data_path / output_name))

    def compute_metrics(self, p):
        """Вычисление метрик для тренировки и валидации."""
        predictions, labels = p
        loss = np.mean(np.square(predictions - labels))  # Пример вычисления потерь
        self.val_losses.append(loss)  # Сохранение потерь валидации
        return {'loss': loss}

    def plot_metrics(self):
        """Построение графиков потерь."""
        epochs = range(len(self.train_losses))

        plt.figure(figsize=(14, 5))

        # График потерь
        plt.subplot(1, 1, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, self.val_losses, label='Validation Loss', color='orange')
        plt.title('Loss during Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

if __name__ == "__main__":
    fine_tuner = FineTuner()
    fine_tuner.fine_tune('train_data.txt', 'val_data.txt')  # Передаем пути к текстовым файлам
    fine_tuner.plot_metrics()  # Вызов для отображения графиков
    print("Модель успешно дообучена и сохранена!")