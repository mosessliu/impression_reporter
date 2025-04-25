FROM dustynv/nanoowl:r36.2.0

COPY app/ /app/

WORKDIR /app

RUN python3 setup.py install
RUN cp /opt/nanoowl/data/owl_image_encoder_patch32.engine src/owl_image_encoder_patch32.engine
RUN rm -rf /root/.cache/clip
