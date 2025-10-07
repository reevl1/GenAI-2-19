from main import generate_ans, TOKENIZER, MODEL
import ast # parse

def test_code_syntax(generated_code):
    if generated_code is None:
        print("функция генерации вернула None")
        return 0

    try:
        ast.parse(generated_code)
    except SyntaxError:
        return 0
    return 1

def test_exec(generated_code):
    try:
        exec(generated_code, {})
        print("код успешно выполнен\n")
        return 1
    except Exception as e:
        print(f"ошибка при выполнении кода: {e}\n")
        return 0

def tests():
    print("\n---------------- Начало тестов ----------------\n")
    prompts = [
        "напиши функцию, которая выводит 'hello, world'",
        "напиши функцию, которая считает факториал",
        "напиши функцию, которая проверяет число на простоту"
    ]
    for p in prompts:
        print(f"промпт: {p}")
        code = generate_ans(p)
        assert test_code_syntax(code)
        assert test_exec(code)
    print("---------------- Все тесты пройдены успешно ----------------\n")

tests()