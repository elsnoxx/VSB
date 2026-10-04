
SET SERVEROUTPUT ON;
DECLARE
    ddl_qry     VARCHAR2 (100);
BEGIN

    ddl_qry := 'DROP TABLE Sickness'; 
    EXECUTE IMMEDIATE ddl_qry;  
    
    ddl_qry :='DROP TABLE Payments'; 
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE WHours'; 
    EXECUTE IMMEDIATE ddl_qry; 
          
    ddl_qry := 'DROP TABLE Factors';    
    EXECUTE IMMEDIATE ddl_qry;    
    
    ddl_qry := 'DROP TABLE Factor';    
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE Checks';
    EXECUTE IMMEDIATE ddl_qry;  
    
    ddl_qry := 'DROP TABLE "CHECK"';
    EXECUTE IMMEDIATE ddl_qry;  
    
    ddl_qry := 'DROP TABLE TRAINING_HISTORY'; 
    EXECUTE IMMEDIATE ddl_qry; 
      
    ddl_qry := 'DROP TABLE Training'; 
    EXECUTE IMMEDIATE ddl_qry;  
        
    ddl_qry := 'DROP TABLE PPEs';
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE PPE'; 
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE Manager';
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE worker';
    EXECUTE IMMEDIATE ddl_qry; 

    ddl_qry := 'DROP TABLE PI';               
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE Marital_status';      
    EXECUTE IMMEDIATE ddl_qry;  
    
    ddl_qry := 'DROP TABLE EDUCATION';            
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE Dl';                   
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE Disclass';              
    EXECUTE IMMEDIATE ddl_qry;  
    
    ddl_qry := 'DROP TABLE salary_change';             
    EXECUTE IMMEDIATE ddl_qry; 
         
    ddl_qry := 'DROP TABLE METHODS';              
    EXECUTE IMMEDIATE ddl_qry;  
    
    ddl_qry := 'DROP TABLE ACCOUNT';              
    EXECUTE IMMEDIATE ddl_qry; 
       
    ddl_qry := 'DROP TABLE employee_address';              
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE Employee';            
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE address';               
    EXECUTE IMMEDIATE ddl_qry;  
    
    ddl_qry := 'DROP TABLE CITY';             
    EXECUTE IMMEDIATE ddl_qry; 
    
    ddl_qry := 'DROP TABLE COUNTRY';              
    EXECUTE IMMEDIATE ddl_qry; 
    
END;



