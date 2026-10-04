# Project VIS

## Databse setup

Via usin docker command:
```bash
docker run -d --name vis-mariadb -p 3306:3306 -e MARIADB_ROOT_PASSWORD=secret -e MARIADB_DATABASE=vis -e MARIADB_USER=vis_user -e MARIADB_PASSWORD=vis_pass mariadb:11
```


# Used

### Table Data Gateway

An object that acts as a gateway to a database table. One instance handles all the rows in the table.

- MariaDbDeviceRepository
- MariaDbLocationRepository

**Každý repository:**

- reprezentuje jednu tabulku
- obsahuje SQL (SELECT / INSERT / UPDATE / DELETE)
- neobsahuje doménovou logiku

### Transaction Script

Organizes business logic by procedures where each procedure handles a single request.

Tvůj kód:

- DeviceService
- LocationService

**Každá metoda:**

- řeší jeden use-case
- nemáš bohatý doménový model 
- logika je procedurální

### Service Layer

Defines an application's boundary with a layer of services.

### Unit of Work

Maintains a list of objects affected by a business transaction