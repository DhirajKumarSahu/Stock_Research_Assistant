FROM python:3.10-slim-bookworm

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY api.py .

COPY research_assistant.py .

EXPOSE 8080

ENTRYPOINT ["python3", "api.py"]