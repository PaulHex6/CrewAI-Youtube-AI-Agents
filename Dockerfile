FROM python:latest

WORKDIR /usr/src/app

COPY requirements* .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8500

CMD ["streamlit", "run", "app.py"]