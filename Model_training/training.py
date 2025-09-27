
""" Основной сценарий обучения """


from transformers import (
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

from .datasets import prepare_data, create_text_datasets
from .tokenization import initialize_tokenizer
from .metrics import compute_metrics
from .plotting import plot_metrics


class FineTuner:
    def __init__(self, model_name='gpt2', cache_dir='./model_cache/'):
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name, cache_dir=cache_dir)
        self.tokenizer = initialize_tokenizer(model_name, cache_dir)
        self.train_losses = []
        self.val_losses = []

    def fine_tune(self, csv_path, output_name='fine_tuned_model', **kwargs):
        train_data, val_data = prepare_data(csv_path)
        train_dataset, val_dataset = create_text_datasets(train_data, val_data)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=f'{output_name}',
            overwrite_output_dir=True,
            num_train_epochs=kwargs.get('num_train_epochs', 10),
            per_device_train_batch_size=kwargs.get('batch_size', 2),
            learning_rate=kwargs.get('learning_rate', 5e-5),
            save_steps=kwargs.get('save_steps', 1000),
            logging_dir=f'{output_name}/logs/',
            logging_steps=100,
            evaluation_strategy="epoch",
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        self.train_losses = trainer.state.log_history
        self.val_losses = trainer.evaluate()['eval_loss']

        # Сохранение обученной модели и токенизатора
        self.model.save_pretrained(output_name)
        self.tokenizer.save_pretrained(output_name)

    def plot_metrics(self):
        """Отображает графики потерь."""
        plot_metrics(self.train_losses, self.val_losses)
