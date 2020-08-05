FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev build-essential && \
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.4/index.html && \
    pip install -r requirements.txt

WORKDIR /code
COPY . .

CMD ["/bin/bash"]