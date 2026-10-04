-- 1
-- Vytvorit tabulku Statistics, kde events_count nesmi byt mensi nez 0
-- Potom naplnit tabulku IDckama uzivatelu a poctem jejich eventu

CREATE TABLE test."statistics"
(
    person_id int NOT NULL UNIQUE,
    events_count int NOT NULL DEFAULT 0,
    FOREIGN KEY (person_id) REFERENCES test.person(pID),
    CONSTRAINT ch_count_not_negative CHECK (events_count >= 0)
);

INSERT INTO test."statistics" (person_id, events_count)
    SELECT p.pID, count(e.eID)
    from test.person p
             JOIN test.device_event e ON p.pID = e.pID
    GROUP BY p.pID, e.pID

-- 2
-- Smazat zarizeni, ktere nemaji senzor
-- Taky smazat tyto zarizeni z dalsich tabulek (tady konkretne device_event)

DELETE FROM test.device_event
WHERE dID IN (
    SELECT DISTINCT dID
    FROM test.device
    WHERE has_sensor = 0
)

DELETE FROM test.device
WHERE has_sensor = 0

-- 3
-- Nastavit sloupec startDate jako povinny a nastavit defaultni hodnotu na aktualni datum
-- Predtim je treba updatovat radky, kde je startDate = null

UPDATE test.device_event
SET startDate = GETDATE()
WHERE startDate IS NULL

ALTER TABLE test.device_event
ALTER COLUMN
    startDate datetime NOT NULL

ALTER TABLE test.device_event
ADD CONSTRAINT dc_default_current_date
DEFAULT GETDATE() FOR startDate

-- 4
-- Nastavit bossID na null u zamestnancu, kteri meli v poslednich 3 letech vyrizeno alespon 5 eventu

UPDATE test.person
SET bossID = null
FROM (
         SELECT p.pID as id, count(e.eID) as pocet
         from test.person p
                  JOIN test.device_event e ON p.pID = e.pID
         WHERE e.endDate <= DATEADD(year, -3, GETDATE())
         GROUP BY p.pID, e.pID
     ) as zam
WHERE pID = zam.id AND zam.pocet >= 5
