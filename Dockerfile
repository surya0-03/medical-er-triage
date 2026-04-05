FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi==0.111.0 uvicorn==0.30.0 pydantic==2.7.0 numpy==1.26.4 openai==1.12.0 requests
EXPOSE 7860
CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
