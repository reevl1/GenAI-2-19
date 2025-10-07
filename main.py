import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR) # чтобы скрыть предупреждения

TOKENIZER = None
MODEL = None

def get_model():
    """загружает модель и токенизатор, если они еще не были загружены"""

    global TOKENIZER, MODEL

    if TOKENIZER is None or MODEL is None:
        print("Начало загрузки модели и токенизатора...\n")

        TOKENIZER = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        MODEL = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
        TOKENIZER.pad_token_id = TOKENIZER.eos_token_id

        print("Модель и токенизатор загружены.\n")

    return TOKENIZER, MODEL

def validate_inputs(user_prompt, temp, max_len, tokenizer, model):
    """валидация входных данных

    Args:
        user_prompt (str): промпт пользователя
        temp (float):      температура генерации
        max_len (int):     максимальна ядлина ответа 
        tokenizer:         токенизатор
        model:             модель

    Raises:
        ValueError: какой либо входной аргумент некорректен
    """

    if not isinstance(user_prompt, str) or not user_prompt.strip(): raise ValueError("пользовательский промпт должен быть не пустой строкой")
    if not (0.0 < temp <= 2.0): raise ValueError("температура должна быть между 0 и 2")
    if max_len > model.config.max_position_embeddings: raise ValueError(f"максимальная длины ответа превышает допустимы йлимит: {model.config.max_position_embeddings}")
    # обработка слишком длинных промптов
    input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids
    if input_ids.shape[1] > model.config.max_position_embeddings: raise ValueError("cлишком длинный пользовательский промпт")

def generate_ans(user_prompt, tokenizer=None, model=None, max_len=512, num_ret_sequences=1, temp=0.1, do_samp=True, top_p=0.95, top_k=40, repetit_penalty=1.1):
    """генерирует код на основе текстового ввода пользователя

    Args:
        user_prompt (str):       пользовательский промпт
        tokenizer:               токенизатор (опционально)
        model:                   модель (опционально)
        max_len (int):           максимальная длина ответа
        num_ret_sequences (int): количество генерируемых последовательностей
        temp (float):            температура генерации
        do_samp (bool):          использовать ли сэмплирование
        top_p (float):           вероятность для nucleus sampling - выбор минимально возможного числа токенов, чья суммарная вероятность превышает порог p
        top_k (int):             количество токенов для top-k sampling - модель рассматривает только k наиболее вероятных токенов
        repetit_penalty (float): штраф за повторения 

    Returns:
        str | None: сгенерированный код или None в случае ошибки.
    """

    try:
        if tokenizer is None or model is None: tokenizer, model = get_model()

        validate_inputs(user_prompt, temp, max_len, tokenizer, model)

        # модель, которую необходимо использовать, продолжает текст, а не выдает самостоятельный ответ (как например chatgpt)
        # поэтому пользовательский промпт заворачивается в питоновский комментарий, а далее напишется "def", чтобы модель восприняла это
        # как начало кода, который необходимо продолжить.
        text = f"'''\nuser prompt:\n{user_prompt.strip()}\nWrite Python code IN ENGLISH below:\n'''\ndef"

        input_ids = tokenizer(text, return_tensors="pt")

        generated_ids = model.generate(
            **input_ids,
            max_length=max_len,
            num_return_sequences=num_ret_sequences,
            temperature=temp,
            do_sample=do_samp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetit_penalty,
            pad_token_id=tokenizer.eos_token_id 
        )

        ans = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # обрезаем начало (пользовательский промпт)
        # также модель часто начинает повторяться, если не достигла максимальной длины ответа.
        # поэтому все повторения (они опять же начинаются с плльзовательского промпта) обрезаем тем же срезом.
        ans = ans[len(text) - 3:ans.find(text[:15], len(text)) if ans.find(text[:15], len(text)) != -1 else len(ans)]
        
        print(f"Ответ модели:\n\n{ans}\n")
        return ans

    except Exception as e:
        print(f"ОШИБКА: {e}")
        return None