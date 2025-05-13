Admin username: sa
Password: AdminTest

Running docker file

build container -> docker build -t djangoprojekt .
start container -> docker run -d -p 8000:8000 --name djangoprojekt_container djangoprojekt

stop container -> docker stop djangoprojekt_container
remove container -> docker rm djangoprojekt_container


elsnoxx
Benjamin1*