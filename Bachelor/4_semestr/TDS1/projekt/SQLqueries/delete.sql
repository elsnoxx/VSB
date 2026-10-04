-- Drop all tables with error handling
BEGIN
  FOR table_rec IN (SELECT table_name FROM user_tables WHERE table_name IN ('SERVICEUSAGE', 'FEEDBACK', 'SERVICEPRICEHISTORY', 'ROOMTYPEPRICEHISTORY', 'RESERVATION', 'PAYMENT', 'SERVICE', 'ROOM', 'ROOMTYPE', 'GUEST', 'EMPLOYEE', 'GUESTARCHIVE', 'INVOICES', 'EVENTLOG', 'TEXTEXAMPLE', 'NUMBEREXAMPLE', 'BLOBEXAMPLE')) LOOP
    BEGIN
      EXECUTE IMMEDIATE 'DROP TABLE ' || table_rec.table_name || ' CASCADE CONSTRAINTS';
      DBMS_OUTPUT.PUT_LINE('Dropped table: ' || table_rec.table_name);
    EXCEPTION
      WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error dropping table ' || table_rec.table_name || ': ' || SQLERRM);
    END;
  END LOOP;
END;
/

-- Drop all views with error handling
BEGIN
  FOR view_rec IN (SELECT view_name FROM user_views WHERE view_name IN ('VIP_GUESTS_VIEW', 'AVAILABLE_ROOMS', 'FUTURE_BOOKINGS', 'STANDARD_GUESTS', 'PAYMENT_SUMMARY')) LOOP
    BEGIN
      EXECUTE IMMEDIATE 'DROP VIEW ' || view_rec.view_name;
      DBMS_OUTPUT.PUT_LINE('Dropped view: ' || view_rec.view_name);
    EXCEPTION
      WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error dropping view ' || view_rec.view_name || ': ' || SQLERRM);
    END;
  END LOOP;
END;
/

-- Drop all sequences with error handling
BEGIN
  FOR seq_rec IN (SELECT sequence_name FROM user_sequences WHERE sequence_name IN ('SEQ_INVOICE_ID', 'SEKVENCE2')) LOOP
    BEGIN
      EXECUTE IMMEDIATE 'DROP SEQUENCE ' || seq_rec.sequence_name;
      DBMS_OUTPUT.PUT_LINE('Dropped sequence: ' || seq_rec.sequence_name);
    EXCEPTION
      WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error dropping sequence ' || seq_rec.sequence_name || ': ' || SQLERRM);
    END;
  END LOOP;
END;
/

-- Drop all synonyms with error handling
BEGIN
  FOR syn_rec IN (SELECT synonym_name FROM user_synonyms WHERE synonym_name IN ('ACTIVE_RESERVATIONS')) LOOP
    BEGIN
      EXECUTE IMMEDIATE 'DROP SYNONYM ' || syn_rec.synonym_name;
      DBMS_OUTPUT.PUT_LINE('Dropped synonym: ' || syn_rec.synonym_name);
    EXCEPTION
      WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error dropping synonym ' || syn_rec.synonym_name || ': ' || SQLERRM);
    END;
  END LOOP;
END;
/

-- Drop all indexes with error handling
BEGIN
  FOR idx_rec IN (SELECT index_name FROM user_indexes WHERE index_name IN ('IDX_GUEST_EMAIL', 'IDX_ROOM_TYPE_NUMBER') AND table_owner = USER) LOOP
    BEGIN
      EXECUTE IMMEDIATE 'DROP INDEX ' || idx_rec.index_name;
      DBMS_OUTPUT.PUT_LINE('Dropped index: ' || idx_rec.index_name);
    EXCEPTION
      WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error dropping index ' || idx_rec.index_name || ': ' || SQLERRM);
    END;
  END LOOP;
END;
/

-- Reset IDENTITY columns (for future table creation)
-- Note: Oracle automatically resets IDENTITY sequences when table is dropped
-- But if you want to manually reset existing sequences, you can use:

ALTER TABLE Guest MODIFY guest_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE Employee MODIFY employee_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE RoomType MODIFY room_type_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE Room MODIFY room_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE Payment MODIFY payment_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE Reservation MODIFY reservation_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE Service MODIFY service_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE ServiceUsage MODIFY usage_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE Feedback MODIFY feedback_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE ServicePriceHistory MODIFY sph_id GENERATED ALWAYS AS IDENTITY (START WITH 1);
ALTER TABLE RoomTypePriceHistory MODIFY rtph_id GENERATED ALWAYS AS IDENTITY (START WITH 1);

DBMS_OUTPUT.PUT_LINE('Database cleanup completed.');