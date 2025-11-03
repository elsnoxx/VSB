-- ===============================================
-- TDS2 - Lekce 11
-- Perzistentní stav proměnných v PACKAGE
-- Využití Oracle balíčků
-- ===============================================

-- ===============================================
-- 1. PACKAGE s perzistentními proměnnými
-- ===============================================

-- a
CREATE OR REPLACE PACKAGE MIK0486.global_package AS
    g_global_variable VARCHAR2(100);
    PROCEDURE set_global_variable(p_value VARCHAR2);
    FUNCTION get_global_variable RETURN VARCHAR2;
END;

CREATE OR REPLACE PACKAGE BODY MIK0486.global_package AS
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