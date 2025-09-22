

Docker for testing sql databese

docker run -d --name oracle-xe -p 1521:1521 -p 5500:5500 -e ORACLE_PASSWORD=MyPassword123 gvenzl/oracle-xe
