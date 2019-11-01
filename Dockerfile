FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN mkdir /app

WORKDIR /app/

ENV LANG C.UTF-8

# apt-get
RUN apt-get update
RUN apt-get -y install wget
RUN apt-get install -y build-essential \
  gcc make yasm autoconf curl \
  automake cmake libtool \
  checkinstall libmp3lame-dev \
  pkg-config libunwind-dev \
  zlib1g-dev libssl-dev \
  aptitude libsndfile-dev 

## ffmpeg
RUN wget https://www.ffmpeg.org/releases/ffmpeg-4.0.2.tar.gz
RUN tar -xzf ffmpeg-4.0.2.tar.gz; rm -r ffmpeg-4.0.2.tar.gz
# FIXME
RUN cd ./ffmpeg-4.0.2; ./configure --enable-libmp3lame --enable-decoder=mjpeg,png --enable-encoder=png --enable-openssl
RUN cd ./ffmpeg-4.0.2; make
RUN  cd ./ffmpeg-4.0.2; make install


ADD requirements.txt /app/

# pip packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# dockerize
ENV DOCKERIZE_VERSION v0.6.1
RUN wget https://s3.ap-northeast-2.amazonaws.com/mindlogic-dev-packages/dockerize-linux-amd64-v0.6.1.tar.gz \
  && tar -C /usr/local/bin -xzvf dockerize-linux-amd64-v0.6.1.tar.gz \
  && rm dockerize-linux-amd64-v0.6.1.tar.gz

RUN echo export PATH="$HOME/.local/bin:$PATH"

EXPOSE 6001
EXPOSE 6002
EXPOSE 6003
EXPOSE 6004

COPY . /app

CMD ["sleep", "infinity"]