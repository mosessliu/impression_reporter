FROM dustynv/nanoowl:r36.2.0

COPY app/ /app/

WORKDIR /app

RUN python3 setup.py install
