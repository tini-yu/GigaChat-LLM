FROM python:3.12

WORKDIR /llm

COPY ./requirements.txt /llm/requirements.txt

RUN pip install --no-cache-dir --upgrade --default-timeout=10000 -r /llm/requirements.txt

COPY . /llm

EXPOSE 8000

CMD ["python", "main.py"]