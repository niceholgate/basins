docker build -t basins .

docker run -it --mount source=basins-images,target=/images basins
