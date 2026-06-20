FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    prometheus-client \
    python-multipart \
    transformers \
    peft \
    pyyaml

COPY inference/ ./inference/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
