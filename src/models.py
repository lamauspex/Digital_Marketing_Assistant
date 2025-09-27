
""" Модели и токенизаторы """


from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        is_decoder=True
    )
    return model, tokenizer
