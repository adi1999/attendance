FROM jhonatans01/python-dlib-opencv
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python /app/app.py