FROM tensorflow/tensorflow:2.3.1

RUN mkdir -p /recap_service
WORKDIR /recap_service

COPY requirements.txt ./
RUN pip3 install --proxy $HTTP_PROXY uvloop==0.14.0
RUN pip3 install --proxy $HTTP_PROXY -r requirements.txt

COPY src/ .

EXPOSE 8000

CMD ["uvicorn", "--host", "0.0.0.0", "main_app:app"]
