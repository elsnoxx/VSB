docker run --detach --name parkinglot --env MARIADB_ROOT_PASSWORD=myparkinglot -p 3306:3306 mariadb:latest


user----> webapi
pass----> wabapplogin

popis api na -> /ApiDescription


Odpovím za chabra dyk 

Chtěl na začátku proste vidět co to umí ta cela apka,

Pak chce vidět validaci u formulářů jak na webu tak i desktop,

pak dokumentace k jsonu(bacha na return type nemel bys tam mit jen Task1) ale treba object{name,id….} tady tohle skoro nikdo nemel a bral zbytecne body,

Pak chtěl vidět jak se volá api na desktop aplikaci (bacha na using u Httpclientu),
chtěl taky vidět jestli máš nějaký speciál api klíč aby si to api nemohl volat každý(taky skoro nikdo neměl xd),
A nakonec jak řešit autorizace admina,
a ještě jak se dělala databáze co jste použili dapper treba nebo custom … nic jiného neřešil,

Hodnotí i celou apku jako takovou ale videl jsem ze to dal vsem full za tu část takze zbytek nehrotí


docker build parking lot webapp

1. docker build -t parkinglotweb .

2. docker run -p 8080:8080 --name parkinglotweb_container parkinglotweb