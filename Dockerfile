FROM python:3.10
ADD src/ basins/src/
ADD requirements.txt basins/
ADD config.py basins/src/
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y ffmpeg
RUN pip install -r basins/requirements.txt
CMD ["python", "basins/src/basins.py"]