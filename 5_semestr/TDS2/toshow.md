# TDS Project - Hotel Management System

This project demonstrates various Oracle SQL and PL/SQL concepts including triggers, dynamic SQL, procedures, functions, exceptions, and cursors.

## Demonstration Examples

### 1. Trigger instead of
Ukažte z Úkol 13 (lesson13.sql):

```sql
-- Spuštění triggerů
BEGIN
    test_triggers;
END;
```

### 2. Dynamické SQL
Ukažte z Úkol 12:

```sql
-- Test dynamického SQL
demo_guest_dynamic_sql;
```

### 3. Procedura s IN/OUT parametry
Ukažte z Úkol 8:

```sql
-- Test procedury s parametry
DECLARE
    v_fullname VARCHAR2(200);
BEGIN
    guest_info_param(p_guest_id => 1, p_fullname => v_fullname);
    DBMS_OUTPUT.PUT_LINE('Returned fullname: ' || v_fullname);
END;
```

### 4. Přetěžování funkce
Ukažte z Úkol 10:

```sql
-- Test overloadingu
DBMS_OUTPUT.PUT_LINE('2 params: "' || string_pkg.reverse_concat('Hello', 'World') || '"');
DBMS_OUTPUT.PUT_LINE('3 params: "' || string_pkg.reverse_concat('A', 'B', 'C') || '"');
```

### 5. Vlastní výjimka
Ukažte z Úkol 7:

```sql
-- Test vlastních výjimek
guest_info_custom_exception_msg2; -- RAISE_APPLICATION_ERROR
```

### 6. Kurzor
Ukažte z Úkol 5:

```sql
-- Test kurzoru
CHECK_GUEST_FOR_LOOP; -- Kurzor FOR LOOP
```

