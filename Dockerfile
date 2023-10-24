FROM nvcr.io/nvidia/tritonserver:23.08-pyt-python-py3 as tritonserver

ENV TS_VERSION=23.08

WORKDIR /app/inference_triton

COPY ./trocr-captcha-killer ./trocr-captcha-killer
COPY src/model.py model_repository/add_sub/1/model.py
COPY src/config.pbtxt model_repository/add_sub/config.pbtxt
COPY src/utils_data.py model_repository/add_sub/1/utils_data.py
COPY src/requirements_server.txt requirements.txt

RUN pip install -r requirements.txt

CMD tritonserver --model-repository `pwd`/model_repository --backend-config=python,shm-default-byte-size=256777216

FROM nvcr.io/nvidia/tritonserver:23.08-py3-sdk as tritonclient

WORKDIR /app/client

COPY ./src/client.py client.py
COPY ./pyproject.toml pyproject.toml
COPY ./data/input/test_example.json test_example.json

RUN pip install poetry && \ 
    poetry install --only tritonclient && \
    pip install tritonclient

CMD python3 client.py
