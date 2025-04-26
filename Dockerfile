FROM dustynv/nanoowl:r36.2.0

RUN rm -rf /root/.cache/clip

COPY app/ /app/

WORKDIR /app

RUN python3 setup.py install
