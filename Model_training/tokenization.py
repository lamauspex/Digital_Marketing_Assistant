
""" Токенизация данных """


from transformers import GPT2Tokenizer


def initialize_tokenizer(model_name='gpt2', cache_dir=None):
    """Инициализирует токенизатор."""
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    return tokenizer
