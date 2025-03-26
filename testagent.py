import time
from pydantic import BaseModel
import random
import logging

logger = logging.getLogger(__name__)

# class Question(BaseModel):
#     # preferences: str
#     location: str

jokes = [
    "Заходит гелий в бар и заказывает пиво. Бармен: 'Извините, но мы не обслуживаем благородных' Гелий не реагирует.",
    "Заходит как-то мужик в бар с говном на руках и говорит: Мужики! Смотрите! Чуть не наступил",
    "Заходит как-то бармен в бар, а он ему как раз.",
    "Заходят в бар бесконечное число математиков. Первый заказывает кружку пива. Второй – половину кружки. Третий – четверть. Вот дурни! – говорит бармен и наливает две кружки пива.",
    "Заходят как-то в бар американец, русский и чукча, а бармен им и говорит: 'Минуточку, я что, в анекдоте?'",
    "Заходят как-то в паб англичанин, ирландец и американец. Англичанин говорит «мне одно пиво', ирландец говорит 'мне два пива', а американец говорит 'ребят, а почему мы на русском разговариваем?'",
    "Бармен: путешественникам во времени не наливаем. Путешественник во времени заходит в бар."
    ]

def get_answer(question: dict):

    # input_json = question.model_dump()
    # # preferences = input_json['preferences']
    # location = input_json['location']
    # response = location

    adr = ""
    for key in question:
        value = question[key]
        if value != "":
            adr += f"{value} "

    logger.info(f"Address: {adr}; JSON:\n\t{question}")

    if adr != "":
        response = f"Выбранное место: {adr}. {jokes[random.randint(0, 6)]}"    
    else:
        response = "ОШИБКА: Недостаточно данных о выбранном месте (пустой адрес)"
        logger.warning("Empty address")
    
    sleep = random.randint(3, 4)
    time.sleep(sleep)
    return response