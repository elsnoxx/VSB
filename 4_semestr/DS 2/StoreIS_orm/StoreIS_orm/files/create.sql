-- Store Database, Database System II
-- May, 7, 2024
-- db.cs.vsb.cz

create table Product (
    id_product INTEGER NOT NULL primary key,
    name       VARCHAR(100) NOT NULL,
    unit_price FLOAT NOT NULL
);

create table ProductPrice (
    id_product INTEGER NOT NULL FOREIGN KEY REFERENCES Product(id_product),
    unit_price FLOAT NOT NULL,
    "from"     DATE NOT NULL,
    "to"       DATE NOT NULL
);

create table Staff (
    id_staff INTEGER NOT NULL primary key,
    lname    VARCHAR(30) NOT NULL,
    fname    VARCHAR(30) NOT NULL,
    "from"   DATE,
    "to"     DATE,
    address  VARCHAR(100)
);

create table Supply (
    id_supply  INTEGER NOT NULL primary key,
    id_product INTEGER NOT NULL FOREIGN KEY REFERENCES Product(id_product),
    unit_price FLOAT NOT NULL,
    quantity   FLOAT NOT NULL,
    "date"     DATE NOT NULL
);

create table "User" (
    id_user INTEGER NOT NULL primary key,
    fname   VARCHAR(30) NOT NULL,
    lname   VARCHAR(30) NOT NULL,
    address VARCHAR(100),
    "level" INTEGER NOT NULL
);

create table "Order" (
    id_order   INTEGER NOT NULL primary key identity,
    id_user    INTEGER FOREIGN KEY REFERENCES "User"(id_user),
    id_staff   INTEGER NOT NULL FOREIGN KEY REFERENCES Staff(id_staff),
    date_order DATE NOT NULL,
    price      FLOAT
);

create table OrderItem (
    id_order   INTEGER NOT NULL FOREIGN KEY REFERENCES "Order"(id_order),
    id_product INTEGER NOT NULL FOREIGN KEY REFERENCES Product(id_product),
    unit_price FLOAT NOT NULL,
    quantity   FLOAT NOT NULL,
	primary key (id_order, id_product)
);