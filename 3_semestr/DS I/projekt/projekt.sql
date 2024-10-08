-- Vytvoření tabulky pro restaurace
CREATE TABLE Restaurace (
    RestauraceID INT PRIMARY KEY AUTO_INCREMENT,
    Nazev VARCHAR(100) NOT NULL,
    Adresa VARCHAR(255) NOT NULL,
    Telefon VARCHAR(15),
    OteviraciHodiny VARCHAR(50)
);

-- Vytvoření tabulky pro stoly
CREATE TABLE Stoly (
    StolID INT PRIMARY KEY AUTO_INCREMENT,
    RestauraceID INT,
    CisloStolu INT NOT NULL,
    Kapacita INT NOT NULL,
    Status ENUM('volný', 'obsazený', 'rezervovaný') NOT NULL,
    FOREIGN KEY (RestauraceID) REFERENCES Restaurace(RestauraceID)
);

-- Vytvoření tabulky pro hosty
CREATE TABLE Hosty (
    HostID INT PRIMARY KEY AUTO_INCREMENT,
    Jmeno VARCHAR(50) NOT NULL,
    Prijmeni VARCHAR(50) NOT NULL,
    Telefon VARCHAR(15),
    Email VARCHAR(100),
    Poznamka TEXT
);

-- Vytvoření tabulky pro rezervace
CREATE TABLE Rezervace (
    RezervaceID INT PRIMARY KEY AUTO_INCREMENT,
    StolID INT,
    HostID INT,
    DatumCas DATETIME NOT NULL,
    PocetOsob INT NOT NULL,
    Poznamka TEXT,
    FOREIGN KEY (StolID) REFERENCES Stoly(StolID),
    FOREIGN KEY (HostID) REFERENCES Hosty(HostID)
);

-- Vytvoření tabulky pro jídla (menu)
CREATE TABLE Jidla (
    JidloID INT PRIMARY KEY AUTO_INCREMENT,
    Nazev VARCHAR(100) NOT NULL,
    Popis TEXT,
    Cena DECIMAL(10, 2) NOT NULL,
    Kategorie VARCHAR(50) NOT NULL
);

-- Vytvoření tabulky pro číšníky/servírky
CREATE TABLE Cisnici (
    CisnikID INT PRIMARY KEY AUTO_INCREMENT,
    Jmeno VARCHAR(50) NOT NULL,
    Prijmeni VARCHAR(50) NOT NULL,
    Telefon VARCHAR(15),
    Pozice VARCHAR(50)
);

-- Vytvoření tabulky pro objednávky
CREATE TABLE Objednavky (
    ObjednavkaID INT PRIMARY KEY AUTO_INCREMENT,
    StolID INT,
    HostID INT,
    CisnikID INT,
    DatumCas DATETIME NOT NULL,
    CelkovaCastka DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (StolID) REFERENCES Stoly(StolID),
    FOREIGN KEY (HostID) REFERENCES Hosty(HostID),
    FOREIGN KEY (CisnikID) REFERENCES Cisnici(CisnikID)
);

-- Vytvoření tabulky pro platby
CREATE TABLE Platby (
    PlatbaID INT PRIMARY KEY AUTO_INCREMENT,
    ObjednavkaID INT,
    Castka DECIMAL(10, 2) NOT NULL,
    ZpusobPlatby ENUM('hotově', 'kartou') NOT NULL,
    DatumCas DATETIME NOT NULL,
    FOREIGN KEY (ObjednavkaID) REFERENCES Objednavky(ObjednavkaID)
);

-- Vytvoření vazební tabulky pro objednaná jídla
CREATE TABLE ObjednanaJidla (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    ObjednavkaID INT,
    JidloID INT,
    Mnozstvi INT NOT NULL,
    Cena DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (ObjednavkaID) REFERENCES Objednavky(ObjednavkaID),
    FOREIGN KEY (JidloID) REFERENCES Jidla(JidloID)
);


-- Vložení vzorových dat do tabulky restaurací
INSERT INTO Restaurace (Nazev, Adresa, Telefon, OteviraciHodiny) VALUES
('Restaurace U Šéfa', 'Hlavní 123, Praha', '+420 123 456 789', 'Po-Ne 10:00-22:00'),
('Pizzeria Bella', 'Květinová 456, Brno', '+420 987 654 321', 'Po-Ne 11:00-23:00');

-- Vložení vzorových dat do tabulky stolů
INSERT INTO Stoly (RestauraceID, CisloStolu, Kapacita, Status) VALUES
(1, 1, 4, 'volný'),
(1, 2, 2, 'rezervovaný'),
(1, 3, 6, 'obsazený'),
(2, 1, 4, 'volný'),
(2, 2, 2, 'volný');

-- Vložení vzorových dat do tabulky hostů
INSERT INTO Hosty (Jmeno, Prijmeni, Telefon, Email, Poznamka) VALUES
('Jan', 'Novák', '+420 111 222 333', 'jan.novak@example.com', 'Alergie na ořechy'),
('Petr', 'Svoboda', '+420 444 555 666', 'petr.svoboda@example.com', 'Vegetarián');

-- Vložení vzorových dat do tabulky rezervací
INSERT INTO Rezervace (StolID, HostID, DatumCas, PocetOsob, Poznamka) VALUES
(1, 1, '2024-10-08 18:30:00', 4, 'Žádost o stůl u okna'),
(2, 2, '2024-10-08 19:00:00', 2, NULL);

-- Vložení vzorových dat do tabulky jídel
INSERT INTO Jidla (Nazev, Popis, Cena, Kategorie) VALUES
('Pizza Margherita', 'Klasická pizza s rajčaty a mozzarellou', 150.00, 'hlavní jídla'),
('Caesar Salát', 'Salát s kuřecím masem a parmezánem', 120.00, 'předkrmy'),
('Tiramisu', 'Tradiční italský dezert', 80.00, 'dezerty');

-- Vložení vzorových dat do tabulky číšníků
INSERT INTO Cisnici (Jmeno, Prijmeni, Telefon, Pozice) VALUES
('Alice', 'Kleinová', '+420 777 888 999', 'číšník'),
('Martin', 'Dvořák', '+420 666 555 444', 'servírka');

-- Vložení vzorových dat do tabulky objednávek
INSERT INTO Objednavky (StolID, HostID, CisnikID, DatumCas, CelkovaCastka) VALUES
(3, 1, 1, '2024-10-08 18:45:00', 400.00),
(1, 2, 2, '2024-10-08 19:10:00', 220.00);

-- Vložení vzorových dat do tabulky plateb
INSERT INTO Platby (ObjednavkaID, Castka, ZpusobPlatby, DatumCas) VALUES
(1, 400.00, 'kartou', '2024-10-08 19:00:00'),
(2, 220.00, 'hotově', '2024-10-08 19:15:00');

-- Vložení vzorových dat do tabulky objednaných jídel
INSERT INTO ObjednanaJidla (ObjednavkaID, JidloID, Mnozstvi, Cena) VALUES
(1, 1, 2, 300.00), -- 2x Pizza Margherita
(1, 3, 1, 80.00),  -- 1x Tiramisu
(2, 2, 1, 120.00);  -- 1x Caesar Salát