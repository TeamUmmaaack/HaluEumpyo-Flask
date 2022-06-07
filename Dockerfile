FROM python:3.8.5

WORKDIR /app
COPY . .

# 형태소 분석기 mecab 설치
RUN cd /tmp && \
    wget "https://www.dropbox.com/s/9xls0tgtf3edgns/mecab-0.996-ko-0.9.2.tar.gz?dl=1" && \
    tar zxfv mecab-0.996-ko-0.9.2.tar.gz?dl=1 && \
    cd mecab-0.996-ko-0.9.2 && \
    ./configure && \
    make && \
    make check && \
    make install && \
    ldconfig

RUN cd /tmp && \
    wget "https://www.dropbox.com/s/i8girnk5p80076c/mecab-ko-dic-2.1.1-20180720.tar.gz?dl=1" && \
    apt install -y autoconf && \
    tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz?dl=1 && \
    cd mecab-ko-dic-2.1.1-20180720 && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    ldconfig

# 형태소 분석기 mecab 파이썬 패키지 설치
RUN cd /tmp && \
    git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git && \
    cd mecab-python-0.996 && \
    python setup.py build && \
    python setup.py install

# Cleaning
RUN apt-get remove -y build-essential && \
    rm -rf mecab-*


RUN pip install --upgrade pip
RUN pip install -r requirements.txt && \
    rm *requirements.txt && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 5000

CMD python ./app.py