FROM python:3.10

ADD src/ basins/src/
ADD requirements.txt basins/

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y ffmpeg --fix-missing
RUN pip install -r basins/requirements.txt

EXPOSE 8000

WORKDIR /basins
CMD [ "uvicorn", "src.main:app", "--host=0.0.0.0", "--port=8000"]
