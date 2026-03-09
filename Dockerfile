FROM pytorch/pytorch

WORKDIR /prj

COPY ./requirements_docker.txt /prj/requirements.txt
RUN pip install --no-cache-dir -r /prj/requirements.txt

COPY ./src /prj/src
COPY ./data/test /prj/data/test
COPY ./model-onnx /prj/model-onnx

CMD ["fastapi", "run", "src/app.py", "--port", "8000"]
