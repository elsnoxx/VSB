CREATE OR REPLACE FUNCTION CreateArticleYear (
    p_year IN NUMBER
) RETURN NUMBER IS
    v_count NUMBER;
BEGIN
    -- Create the new table z_article_year
    EXECUTE IMMEDIATE '
        CREATE TABLE z_article_year (
            aid INT,
            jid INT NULL,
            UT_WoS VARCHAR2(25) NULL,
            name NVARCHAR2(1000) NOT NULL,
            type NVARCHAR2(40) NULL,
            year INT NOT NULL,
            author_count INT NULL,
            institution_count INT NULL,
            last_update DATE NOT NULL,
            PRIMARY KEY (aid),
            FOREIGN KEY (jid) REFERENCES z_journal
        )
    ';

    -- Copy records from z_article to z_article_year
    EXECUTE IMMEDIATE '
        INSERT INTO z_article_year (aid, jid, UT_WoS, name, type, year, author_count, institution_count, last_update)
        SELECT aid, jid, UT_WoS, name, type, year, author_count, institution_count, SYSDATE
        FROM z_article
        WHERE year = :1
    ' USING p_year;

    -- Get the count of records in the new table
    EXECUTE IMMEDIATE '
        SELECT COUNT(*)
        FROM z_article_year
    ' INTO v_count;

    -- Drop the new table
    EXECUTE IMMEDIATE 'DROP TABLE z_article_year CASCADE CONSTRAINTS';

    -- Return the count of records
    RETURN v_count;
END;
/
