FROM python:3.8.2

RUN pip install setuptools
COPY requirements.txt /
RUN pip install -r /requirements.txt --no-cache-dir
