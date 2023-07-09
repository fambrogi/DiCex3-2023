FROM python:3.7.9-slim-buster

RUN apt-get update -y && apt-get install -y --no-install-recommends curl python3-pip python3-dev libsm6 libxext6 libxrender-dev libatlas-base-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libopencv-dev build-essential pkg-config libjpeg-dev libpng-dev libgtk-3-dev && rm -rf /var/lib/apt/lists/*

#Allow pi wheels 
RUN echo "[global]\nextra-index-url=https://www.piwheels.org/simple" >> /etc/pip.conf

RUN  mkdir app
RUN  cd  app
WORKDIR  /app

ADD app.py .
ADD requirements.txt .
ADD images/in ./images/in

RUN mkdir images/out
RUN pip3 install --upgrade pip; pip3 install --no-cache-dir -r requirements.txt;
RUN ls

ENV FLASK_APP=app.py

EXPOSE 5000

ENTRYPOINT [ "python3" ]
CMD ["app.py"]
