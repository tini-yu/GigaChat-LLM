import asyncio
import os
import time
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
import logging

log_directory = "./llm_logs"
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, "llm.log")
logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a", format="%(name)s %(asctime)s | %(levelname)s | %(message)s", encoding='utf-8')

import agent2_0
import testagent

app = FastAPI()
load_dotenv()
semaphore = asyncio.Semaphore(10)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Или укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# class Question(BaseModel):
#     # preferences: str
#     location: str

# class Request(BaseModel):
#     info: dict

requestuuid = 0
logger = logging.getLogger(__name__)
logger.info("APP STARTED")

@app.get("/")
def read_root():
    return {"api для запросы к ЛЛМ"}


@app.post("/ask")
async def get_answer(question: dict):
    logger.info(f"Input: {question}")

    #Если запрос находится в очереди больше 300 секунд - возвращает ошибку:
    start_time = time.time()

    async with semaphore:
        wait_time = time.time() - start_time
        logger.info(f"Queue time: {wait_time}")
        if wait_time > 300:
            start_time = time.time()
            logger.error("Time in queue over 300s")
            return {"message": "ОШИБКА: Время ожидания запроса в очереди превысило 300 секунд."}
        
        #Если запрос в течении 120 секунд не обработан, возвращает ошибку:
        try:
            start = time.time()
            # response = await asyncio.wait_for(asyncio.to_thread(agent2_0.get_answer, question), timeout=120)
            response = await asyncio.wait_for(asyncio.to_thread(testagent.get_answer, question), timeout=120)
            
            end = time.time()
            overall_time = round(end-start)
            logger.info(f'It took {overall_time} seconds to finish execution.')
            print('It took {} seconds to finish execution.'.format(round(end-start)))
            return {"message":response}
        except asyncio.TimeoutError:
            logger.error("LLM generation took over 120s")
            return {"message":"ОШИБКА: Время ожидания запроса превысило 120 секунд."}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)
