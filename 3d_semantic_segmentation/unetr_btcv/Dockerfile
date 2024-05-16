FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
RUN pip3 install flask orjson scipy einops monai==1.3.0
WORKDIR /app
COPY . .
CMD ["python3", "app.py"]
