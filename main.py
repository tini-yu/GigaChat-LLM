import asyncio
import os
import time
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel

import agent2_0
import testagent

app = FastAPI()
load_dotenv()
semaphore = asyncio.Semaphore(2)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Или укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    # preferences: str
    location: str

class Request(BaseModel):
    info: dict

@app.get("/")
def read_root():
    return {"Hewwooo!"}


@app.post("/ask")
async def get_answer(question: dict):

    #Если запрос находится в очереди больше 120 секунд - возвращает ошибку:
    start_time = time.time()

    async with semaphore:
        # await asyncio.sleep(1) #Не помогает решить проблему с запросами к апи гигачата
        wait_time = time.time() - start_time
        if wait_time > 120:
            start_time = time.time()
            return {"message": "ОШИБКА: Время ожидания запроса в очереди превысило 120 секунд."}
        
        #Если запрос в течении 30 секунд не обработан, возвращает ошибку:
        try:
            start = time.time()
            # response = await asyncio.wait_for(asyncio.to_thread(agent2_0.get_answer, question), timeout=60)
            response = await asyncio.wait_for(asyncio.to_thread(testagent.get_answer, question), timeout=60)
            
            end = time.time()
            print('It took {} seconds to finish execution.'.format(round(end-start)))
            return {"text":response}
        except asyncio.TimeoutError:
            return {"message":"ОШИБКА: Время ожидания запроса превысило 60 секунд."}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)


"""
POST for JSONs like {"preferences":"bridges, architecture memorials", "location":"Red Square, Moscow"}

fast api receives this JSON from Gin
returns LLM's answer
"""