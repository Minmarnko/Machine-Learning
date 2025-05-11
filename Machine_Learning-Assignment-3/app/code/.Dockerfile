FROM python:3.11.9-bookworm

RUN pip install --upgrade pip

COPY . /root/app
WORKDIR /root/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /root/app/

CMD ["streamlit", "run", "app.py"]
