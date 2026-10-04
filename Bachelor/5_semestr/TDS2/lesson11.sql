-- ===============================================
-- TDS2 - Lekce 11
-- Perzistentní stav proměnných v PACKAGE
-- Využití Oracle balíčků
-- ===============================================

-- ===============================================
-- 1. PACKAGE s perzistentními proměnnými
-- ===============================================

-- a
CREATE OR REPLACE PACKAGE FIC0024.global_package AS
    g_global_variable VARCHAR2(100);
    PROCEDURE set_global_variable(p_value VARCHAR2);
    FUNCTION get_global_variable RETURN VARCHAR2;
END;

CREATE OR REPLACE PACKAGE BODY FIC0024.global_package AS
    PROCEDURE set_global_variable(p_value VARCHAR2) AS
    BEGIN
        DBMS_OUTPUT.PUT_LINE('Setting global variable to: ' || p_value);
        g_global_variable := p_value;
    END set_global_variable;

    FUNCTION get_global_variable RETURN VARCHAR2 AS
    BEGIN
        RETURN g_global_variable;
    END get_global_variable;
END;

BEGIN
    global_package.set_global_variable('Everyone can see');
END;

DECLARE
    result VARCHAR2(100);
BEGIN
    result := global_package.get_global_variable;
    DBMS_OUTPUT.PUT_LINE('Global Variable: ' || result);
END;

-- b
-- No Permissions
BEGIN
    DBMS_SCHEDULER.create_job (
            job_name        => 'test_full_job_definition',
            job_type        => 'PLSQL_BLOCK',
            job_action      => 'BEGIN my_job_procedure; END;',
            start_date      => SYSTIMESTAMP,
            repeat_interval => 'freq=hourly; byminute=0; bysecond=0;',
            enabled         => TRUE);
END;

DECLARE
    v_clob CLOB;
    v_len  NUMBER;
BEGIN
    DBMS_LOB.createtemporary(v_clob, TRUE);

    DBMS_LOB.writeappend(v_clob, LENGTH('ABCDE'), 'ABCDE');

    v_len := DBMS_LOB.getlength(v_clob);
    DBMS_OUTPUT.PUT_LINE('CLOB length = ' || v_len);

    DBMS_LOB.freetemporary(v_clob);
END;
/

DECLARE
    req  UTL_HTTP.req;
    resp UTL_HTTP.resp;
    line VARCHAR2(200);
BEGIN
    req := UTL_HTTP.begin_request('http://example.com');

    resp := UTL_HTTP.get_response(req);

    LOOP
        UTL_HTTP.read_line(resp, line, TRUE);
        DBMS_OUTPUT.PUT_LINE(line);
    END LOOP;

    UTL_HTTP.end_response(resp);
EXCEPTION
    WHEN UTL_HTTP.end_of_body THEN
        DBMS_OUTPUT.PUT_LINE('HTTP request finished.');
END;
/

DECLARE
    v_status INTEGER;
BEGIN
    DBMS_OUTPUT.PUT_LINE('Waiting 2 seconds...');
    v_status := DBMS_LOCK.sleep(2);
    DBMS_OUTPUT.PUT_LINE('Done!');
END;
/
