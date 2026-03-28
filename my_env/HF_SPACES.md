# Hugging Face Spaces Deployment (Docker)

This project is Docker-deployable on Hugging Face Spaces.

## Requirements

- Space SDK: Docker
- Port: 7860
- Space tag: openenv

## Build/Run

The server Dockerfile is at:
- my_env/server/Dockerfile

The container entry command runs:
- uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860

## Notes

- API root object is exported in my_env/app.py as `app`.
- OpenAPI docs will be available on the running Space under `/docs`.
- If running baseline inside the Space, set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` and `OPENAI_INFERENCE_SEED`.
