FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY src/app/ ./src/app/
COPY src/chatbot/ ./src/chatbot/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501
CMD ["streamlit", "run", "src/app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
