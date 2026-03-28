FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r my_env/requirements.txt
EXPOSE 7860
CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
