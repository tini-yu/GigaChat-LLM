from langchain.tools import tool
from langchain_gigachat.chat_models import GigaChat
from pydantic import BaseModel, Field
import requests
from utils.data_caching import load_from_cache, save_to_cache
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain.agents import load_tools
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import os
from typing import List
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# Настройка переменных окружения
os.environ["LANGCHAIN_TRACING_V2"] = "true"
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Также добавьте другие необходимые ключи API
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
OPENTRIPMAP_API_KEY = os.getenv("OPENTRIPMAP_API_KEY")


# Класс результата OSM
class OSMResult(BaseModel):
    name: str = Field(description="Название места")
    address: str = Field(description="Адрес места")
    latitude: float = Field(description="Широта")
    longitude: float = Field(description="Долгота")
    nearby: list[str] = Field(description="Список мест поблизости")
    tags: dict = Field(default_factory=dict, description="Теги объекта")
    message: str = Field(description="Сообщение о результатах поиска")


class PlaceOfInterest(BaseModel):
    name: str = Field(description="Название достопримечательности")
    kinds: str = Field(description="Типы достопримечательности")
    distance: float = Field(description="Расстояние от исходной точки в метрах")
    lat: float = Field(description="Широта")
    lon: float = Field(description="Долгота")


class OpenTripMapResult(BaseModel):
    places: List[PlaceOfInterest] = Field(description="Список найденных достопримечательностей")
    message: str = Field(description="Сообщение о результатах поиска")


@tool
def get_nearby_attractions(latitude: float, longitude: float, radius: int = 1000) -> OpenTripMapResult:
    """
    Получает список достопримечательностей в заданном радиусе от указанных координат используя OpenTripMap.

    Args:
        latitude: широта исходной точки из get_osm_data
        longitude: долгота исходной точки из get_osm_data
        radius: радиус поиска в метрах (по умолчанию 1000)
    """
    API_KEY = OPENTRIPMAP_API_KEY  # Замените на ваш ключ API

    # Проверяем кэш
    cache_key = f"opentripmap_{latitude}_{longitude}_{radius}"
    cached_data = load_from_cache(cache_key)
    if cached_data:
        return OpenTripMapResult(**cached_data)

    base_url = "https://api.opentripmap.com/0.1/en/places/radius"
    params = {
        "radius": radius,
        "lon": longitude,
        "lat": latitude,
        "limit": 10,  # Ограничиваем количество результатов
        "apikey": API_KEY
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        places = []
        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {}).get("coordinates", [])

            if geometry and properties.get("name"):
                place = PlaceOfInterest(
                    name=properties.get("name", "Unknown"),
                    kinds=properties.get("kinds", ""),
                    distance=properties.get("dist", 0),
                    lat=geometry[1],
                    lon=geometry[0]
                )
                places.append(place)

        result = OpenTripMapResult(
            places=places,
            message=f"Найдено {len(places)} достопримечательностей в радиусе {radius} метров"
        )

        # Сохраняем в кэш
        save_to_cache(cache_key, result.model_dump())
        return result

    except requests.exceptions.RequestException as e:
        return OpenTripMapResult(
            places=[],
            message=f"Ошибка при запросе к OpenTripMap: {str(e)}"
        )
@tool
def get_osm_data(place: str) -> OSMResult:
    """Получить информацию о месте и близлежащих объектах из OSM."""
    cache_key = f"osm_{place}"
    cached_data = load_from_cache(cache_key)
    if cached_data:
        print('Using cached_data')
        return OSMResult(**cached_data)

    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place,
        "format": "json",
        "addressdetails": 1,
        "limit": 1
    }
    headers = {"User-Agent": "YourAppName/1.0 (zabydy40@gmail.com)"}

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if not data:
            return OSMResult(name="", address="", latitude=0, longitude=0, nearby=[],tags={},
                             message=f"Место '{place}' не найдено.")

        place_data = data[0]
        name = place_data.get('name', place)
        latitude = float(place_data['lat'])
        longitude = float(place_data['lon'])
        address = place_data.get('display_name', 'Адрес не найден')
        tags = place_data.get('extratags', {})  # Получаем теги объекта


        print('==========================================')
        print('osm_data')
        print(place)
        print('')
        print(data)

        print('==========================================')
        result = OSMResult(
            name=name,
            address=address,
            latitude=latitude,
            longitude=longitude,
            nearby=[],  # Только 5 мест
            tags=tags,
            message=f"Информация о месте '{place}' успешно получена."
        )
        save_to_cache(cache_key, result.model_dump())
        return result

    except requests.exceptions.RequestException as e:
        return OSMResult(name="", address="", latitude=0, longitude=0, nearby=[],tags={},
                         message=f"Ошибка при запросе OSM: {str(e)}")


class InteractiveTourGuide:
    def __init__(self, llm, system_prompt, faiss_index_path=''):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # # Загрузка FAISS-индекса
        # self.faiss_index = FAISS.load_local(
        #     folder_path=faiss_index_path,
        #     embeddings=HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2"),
        #     # embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        #     allow_dangerous_deserialization=True
        # )
        # self.retriever = self.faiss_index.as_retriever(k=5)  # TOP_K=5

        # Инициализация всех инструментов
        self.tavily_search = TavilySearchResults(max_results=2)
        self.wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
        self.wikipedia = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
        )
        self.weather = OpenWeatherMapAPIWrapper()
        self.duckduckgo = DuckDuckGoSearchRun()

        # Объединяем все инструменты
        all_tools = [
                        get_osm_data,
                        get_nearby_attractions,
                        self.tavily_search,
                        self.wikidata,
                        self.wikipedia,
                        self.duckduckgo
                    ] + load_tools(["openweathermap-api"])

        self.agent = create_react_agent(llm, all_tools, state_modifier=system_prompt)
        self.chat_history = []

    # def run(self, user_input: str, preferences: list[str]):
    #     """Обрабатывает пользовательский запрос и формирует экскурсию."""
    #     # Формируем контекст с учетом предпочтений
    #     context = f"""
    #     Место: {user_input}
    #     Предпочтения пользователя: {', '.join(preferences)}
    #     """

    #     self.chat_history.append(HumanMessage(content=context))

    #     try:
    #         # Извлекаем релевантный контекст с помощью ретривера
    #         # retrieved_docs = self.retriever.get_relevant_documents(user_input)
    #         # retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    #         # print('retrieved_text: ', retrieved_text, '\n')
    #         # Получаем предыдущий контекст из памяти
    #         memory_variables = self.memory.load_memory_variables({})
    #         previous_context = memory_variables.get("chat_history", [])

    #         # Объединяем текущий запрос с историей
    #         full_context = previous_context + self.chat_history

    #         # Объединяем с текстами из FAISS-индекса
    #         # full_context.append(HumanMessage(content=f"Контекст из базы знаний:\n{retrieved_text}"))
    #         print('FULL_CONTEXT: ', full_context, '\n')
    #         result = self.agent.invoke({"messages": full_context})

    #         # Сохраняем результат в память и историю
    #         ai_message = result["messages"][-1]
    #         self.chat_history.append(ai_message)
    #         self.memory.save_context(
    #             {"input": context},
    #             {"output": ai_message.content}
    #         )

    #         return ai_message.content

    #     except Exception as e:
    #         return f"Error: {str(e)}"

    # def clear_memory(self):
    #     """Очищает память диалога"""
    #     self.memory.clear()
    #     self.chat_history = []

# SYSTEM_PROMPT = """Ты опытный экскурсовод-эксперт с глубокими знаниями истории, культуры и архитектуры.

# Твои обязанности:
# 1. Создавать увлекательные и информативные экскурсии, используя:
#    - Точные исторические факты из Wikipedia и Wikidata
#    - Актуальную информацию о местоположении из OpenStreetMap и OpenTripMap
#    - Текущую погоду и рекомендации с учетом погодных условий из openweathermap-api
#    - Дополнительную информацию из поисковых систем

# 2. Адаптировать экскурсии под интересы пользователя:
#    - Учитывать указанные предпочтения
#    - Принимать во внимание контекст всего диалога
#    - Предлагать альтернативные маршруты при необходимости

# 3. Структурировать информацию:
#    - Начинать с краткого обзора места
#    - Предоставлять исторический контекст
#    - Описывать архитектурные особенности
#    - Включать интересные факты и легенды
#    - Давать практические рекомендации по посещению

# 4. Обеспечивать интерактивность:
#    - Задавать уточняющие вопросы при необходимости
#    - Предлагать связанные места для посещения
#    - Реагировать на запросы дополнительной информации

# Всегда проверяй достоверность информации через доступные инструменты и указывай источники.
# Если информация противоречива или недостоверна, честно сообщай об этом.
# """

    def run(self, user_input: str):
        """Обрабатывает пользовательский запрос и формирует экскурсию."""
        # Формируем контекст с учетом предпочтений
        context = f"""
        Место: {user_input}
        """

        self.chat_history.append(HumanMessage(content=context))

        try:
            full_context = self.chat_history

            # Объединяем с текстами из FAISS-индекса
            # full_context.append(HumanMessage(content=f"Контекст из базы знаний:\n{retrieved_text}"))
            print('FULL_CONTEXT: ', full_context, '\n')
            result = self.agent.invoke({"messages": full_context})

            # Сохраняем результат в память и историю
            ai_message = result["messages"][-1]
            self.chat_history.append(ai_message)
            self.memory.save_context(
                {"input": context},
                {"output": ai_message.content}
            )

            return ai_message.content

        except Exception as e:
            return f"Error: {str(e)}"

SYSTEM_PROMPT = """Ты опытный экскурсовод-эксперт с глубокими знаниями истории, культуры и архитектуры.

Твои обязанности:
1. Создавать увлекательные и информативные обзоры мест, используя:
   - Точные исторические факты из Wikipedia и Wikidata
   - Актуальную информацию о местоположении из OpenStreetMap и OpenTripMap
   - Текущую погоду и рекомендации с учетом погодных условий из openweathermap-api
   - Дополнительную информацию из поисковых систем

2. Структурировать информацию:
   - Начинать с краткого обзора места
   - Предоставлять исторический контекст
   - Описывать архитектурные особенности
   - Включать интересные факты и легенды
   - Давать практические рекомендации по посещению

Всегда проверяй достоверность информации через доступные инструменты и указывай источники.
Если информация противоречива или недостоверна, честно сообщай об этом.
"""


# === Основной скрипт ===
if __name__ == "__main__":
    # Инициализация LLM (например, GigaChat)
    API_TOKEN = os.getenv("API_TOKEN")  # Укажите токен API
    MODEL_NAME = "GigaChat"
    # Задайте путь к сохранённому FAISS-индексу
    # FAISS_INDEX_PATH = "/faiss"

    # Инициализация клиентов и модели
    model = GigaChat(
        credentials=API_TOKEN,
        scope="GIGACHAT_API_PERS",
        model=MODEL_NAME,
        verify_ssl_certs=False
    )
    # Создание экземпляра гида
    # tour_guide = InteractiveTourGuide(llm=model, system_prompt=SYSTEM_PROMPT, faiss_index_path=FAISS_INDEX_PATH)
    tour_guide = InteractiveTourGuide(llm=model, system_prompt=SYSTEM_PROMPT)

    print("Добро пожаловать в интерактивный гид! Напишите 'выход' для завершения.")
    print("Доступные команды:")
    print("- 'очистить' - очистить историю диалога")
    print("- 'помощь' - показать доступные команды")

    user_preferences = input("Введите ваши предпочтения (через запятую): ").split(",")

    while True:
        user_query = input("\nВведите место или запрос: ").strip().lower()

        if user_query == "выход":
            print("Спасибо за использование гида! До свидания!")
            break
        elif user_query == "очистить":
            tour_guide.clear_memory()
            print("История диалога очищена!")
            continue
        elif user_query == "помощь":
            print("Доступные команды:")
            print("- 'выход' - завершить работу")
            print("- 'очистить' - очистить историю диалога")
            print("- 'помощь' - показать это сообщение")
            continue

        response = tour_guide.run(user_query, user_preferences)
        print("\nОтвет гида:")
        print(response)

class Question(BaseModel):
    location: str

class Request(BaseModel):
    info: dict

def get_answer(question: dict):

    API_TOKEN = os.getenv("API_TOKEN")  # Укажите токен API
    MODEL_NAME = os.getenv("MODEL_NAME")

    # Инициализация клиентов и модели
    model = GigaChat(
        credentials=API_TOKEN,
        scope="GIGACHAT_API_PERS",
        model=MODEL_NAME,
        verify_ssl_certs=False
    )
    # Создание экземпляра гида
    try:
        tour_guide = InteractiveTourGuide(llm=model, system_prompt=SYSTEM_PROMPT)

        # user_query = question
        adr = ""
        for key in question:
            value = question[key]
            if value != "":
                adr += f"{value} "
        logger.info(f"Address: {adr}; JSON:\n\t{question}")
        if adr != "":
            user_query = f"1. Расскажи мне информацию о {adr}. 2. Расскажи об интересных местах вокруг."
            response = tour_guide.run(user_query)
        else:
            response = "ОШИБКА: Недостаточно данных о выбранном месте (пустой адрес)"
            logger.warning("Empty address")
    except:
        return "ОШИБКА при вызове Гигачата"
    if len(response) > 5 and response[0:5] == "Error":
        code = response.split(", ",2)[1]
        return f"ОШИБКА при вызове Гигачата {code}"
    return response