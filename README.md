uvicorn /src/main:app --port=8000 --reload

docker build -t basins .

docker run -p 8000:8000 --mount source=basins-images,target=/basins/images basins
