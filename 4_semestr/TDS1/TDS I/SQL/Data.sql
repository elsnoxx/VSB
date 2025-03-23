
-- COUNTRY --

INSERT INTO country(country_id,name)
VALUES(1,'The Czech republic');

INSERT INTO country(country_id,name)
VALUES(2,'Germany');

INSERT INTO country(country_id,name)
VALUES(3,'France');

INSERT INTO country(country_id,name)
VALUES(4,'Japan');

INSERT INTO country(country_id,name)
VALUES(5,'The UK');

INSERT INTO country(country_id,name)
VALUES(6,'The US');

INSERT INTO country(country_id,name)
VALUES(7,'Australia');

INSERT INTO country(country_id,name)
VALUES(8,'Canada');

INSERT INTO country(country_id,name)
VALUES(9,'South Africa');

INSERT INTO country(country_id,name)
VALUES(10,'Egypt');


-- CITY --

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(1,'Ludge¯ovice',74714,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(2,'Opava',74601,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(3,'Havi¯ov',73601,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(4,'Praha',12808,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(5,'Brno',60110,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(5,'Brno',60110,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(6,'äkvo¯etice',38801,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(7,'Dob¯Ìö',26301,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(8,'Kladno',27201,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(9,'Jedlov·',56991,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(10,'Star˝ Plzenec',33202,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(11,'StudÈnka',74213,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(12,'ätÌtn· nad Vl·¯Ì-Popov',76333,1);

INSERT INTO city(city_id,name, postcode,country_id)
VALUES(13,'Broumov',55001,1);


-- ADDRESS --

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(1,1,'U KapliËky',11);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(2,1,'U Rybniku',10);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(3,3,'DÏlnick·',1132);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(4,4,'U Nemocnice',2);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(5,5,'Jihlavsk·',20);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(6,6,'RokitanskÈho',5);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(7,7,'Konvalinkov·',36);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(8,8,'GrÈgrovo n·mÏstÌ',81);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(9,9,'KopeËek',55);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(10,10,'GruzÌnsk·',1);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(11,11,'Herbenova',13);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(12,4,'Drtinova',12);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(13,12,'SudomÌrky',11);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(14,4,'K Zast·vce',1135);

INSERT INTO address(address_id,city_id,street,"Number")
VALUES(15,13,'Ke Stanici',5);

-- NAPLNÃNÕ Employee --

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (3,'Ellen','Kacalova','+420737254137','ellen.kacalova@msa.com',TO_DATE('06/08/1975', 'DD/MM/YYYY'),'750806/5100','HR Manager',1,
'Kac001',TO_DATE('01/05/2016', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,NULL);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (4,'David','Tesa¯','+420774110765','Tesar.D@gmail.com',TO_DATE('04/04/1970', 'DD/MM/YYYY'),'700404/5321','Policeman',1,
'TES025',TO_DATE('15/12/2010', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,NULL);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (1,'Michal','Kraninger','+420732637540','Kraninger.M@seznam.cz',TO_DATE('06/09/1981', 'DD/MM/YYYY'),'810906/5514','Safety Officer',1,
'KRA028',TO_DATE('12/01/2016', 'DD/MM/YYYY'),TO_DATE(NULL),TO_DATE('12/01/2016', 'DD/MM/YYYY'),3);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (2,'Marketa','Kostkov·','+420603336332','marketa.mk@gmail.com',TO_DATE('03/03/1993', 'DD/MM/YYYY'),'930303/5514','Ecologist',1,
'MAK075',TO_DATE('01/01/2018', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,4);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (5,'Jana','Urbanov·','+420772311565','Jana.Urbanova@hilite.com',TO_DATE('01/01/1995', 'DD/MM/YYYY'),'950101/5321','HR business partner',1,
'URB111',TO_DATE('01/01/2021', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,3);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (10,'Filip','Vondr·Ëek','+420732938874 ','umustafaab@cashbackr.com',TO_DATE('03/01/1998', 'DD/MM/YYYY'),'880103/7743','Finance Director',1,
'VON005',TO_DATE('03/01/2015', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,NULL);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (7,'Eva','KropÌkov·','+420605111021','flittlebeatlebume@eluvit.com',TO_DATE('06/10/1999', 'DD/MM/YYYY'),'995106/8942','Assistant',1,
'KRO002',TO_DATE('02/03/2020', 'DD/MM/YYYY'),TO_DATE('03/10/2021', 'DD/MM/YYYY'),CURRENT_DATE,10);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (8,'Dana','Tvrd·','+420595455795','seif.manal@boosterdomains.tk',TO_DATE('06/11/1992', 'DD/MM/YYYY'),'926106/8681','Teacher',1,
'TVR003',TO_DATE('03/09/2018', 'DD/MM/YYYY'),TO_DATE('31/12/2018', 'DD/MM/YYYY'),CURRENT_DATE,3);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (9,'David','Kraus','+420731736427','4alp.rnsmn@refek.site',TO_DATE('24/04/2001', 'DD/MM/YYYY'),'100424/3004','Driver',0,
'KRA004',TO_DATE('01/06/2020', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,4);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (6,'Petr','MÏöù·k','+420604026283','ranisa.nisa19i@speedan.com',TO_DATE('30/03/2001', 'DD/MM/YYYY'),'010330/7120','Electrician',0,
'MES001',TO_DATE('03/09/2021', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,4);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (11,'PavlÌna','Rudincov·','+420732494559','6roferreirapesso2@cggup.com',TO_DATE('23/12/2000', 'DD/MM/YYYY'),'006223/1390','Accountat',1,
'RUD006',TO_DATE('03/09/2021', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,10);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (12,'Jaroslav','Zan·öka','+420733964561','nfnz.ltdz@umaasa.com',TO_DATE('14/11/1989', 'DD/MM/YYYY'),'891114/8818','Maintenance Eng.',0,
'ZAN007',TO_DATE('03/01/2015', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,3);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (13,'Jan','Zubec','+420602812277','ndevi@readt.site',TO_DATE('26/02/1994', 'DD/MM/YYYY'),'940226/515','Operator',0,
'ZUB008',TO_DATE('02/01/2014', 'DD/MM/YYYY'),TO_DATE('31/12/2019', 'DD/MM/YYYY'),CURRENT_DATE,3);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (14,'Krist˝na','Vagenknechtov·','+420563536635','bsofiane.lov@usayoman.com',TO_DATE('29/01/2001', 'DD/MM/YYYY'),'915129/2904','Warehouse',0,
'VAG009',TO_DATE('10/06/2018', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,4);

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,manager_id)
VALUES (15,'Sabina','»ud·kov·','+420763229057','7btweezy25x@caraparcal.com',TO_DATE('28/4/1991', 'DD/MM/YYYY'),'915428/5342','Purchaser',1,
'CUD010',TO_DATE('10/06/2021', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,4);


-- NAPLNÃNÕ Employee_Address --

INSERT INTO employee_address(employee_id,address_id)
VALUES(1,1);

INSERT INTO employee_address(employee_id,address_id)
VALUES(2,2);

INSERT INTO employee_address(employee_id,address_id)
VALUES(3,3);

INSERT INTO employee_address(employee_id,address_id)
VALUES(4,4);

INSERT INTO employee_address(employee_id,address_id)
VALUES(5,5);

INSERT INTO employee_address(employee_id,address_id)
VALUES(6,6);

INSERT INTO employee_address(employee_id,address_id)
VALUES(7,7);

INSERT INTO employee_address(employee_id,address_id)
VALUES(8,8);

INSERT INTO employee_address(employee_id,address_id)
VALUES(9,9);

INSERT INTO employee_address(employee_id,address_id)
VALUES(10,10);

INSERT INTO employee_address(employee_id,address_id)
VALUES(11,11);

INSERT INTO employee_address(employee_id,address_id)
VALUES(12,12);

INSERT INTO employee_address(employee_id,address_id)
VALUES(13,13);

INSERT INTO employee_address(employee_id,address_id)
VALUES(14,14);

INSERT INTO employee_address(employee_id,address_id)
VALUES(15,15);

-- WORKER ---

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(1,1,1);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(2,1,1);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(5,1,1);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(6,2,0);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(7,1,1);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(8,1,1);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(9,2,0);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(11,1,1);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(12,2,0);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(13,3,0);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(14,3,0);

INSERT INTO  worker (employee_id,category_id,white_collar)
VALUES(15,1,1);

-- MANAGER --

INSERT INTO  Manager (employee_id,pool_car,department,global_position,nfs)
VALUES(3,1,'HR',0,10);

INSERT INTO  Manager (employee_id,pool_car,department,global_position,nfs)
VALUES(4,1,'Police',0,20);

INSERT INTO  Manager (employee_id,pool_car,department,global_position,nfs)
VALUES(10,1,'Finance',1,20);

-- PPE

INSERT INTO PPE(ppe_id,ppe_name,last_update)
VALUES(1,'Safety Shoes',CURRENT_DATE);

INSERT INTO PPE(ppe_id,ppe_name, last_update)
VALUES(2,'Protective Gloves',CURRENT_DATE);

INSERT INTO PPE(ppe_id,ppe_name, last_update)
VALUES(3,'Protective Glasses',CURRENT_DATE);

INSERT INTO PPE(ppe_id,ppe_name, last_update)
VALUES(4,'Safety Helmet',CURRENT_DATE);

INSERT INTO PPE(ppe_id,ppe_name, last_update)
VALUES(5,'Half-face Mask',CURRENT_DATE);

-- PPEs ---

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (1,TO_DATE('12/01/2016', 'DD/MM/YYYY'),TO_DATE('12/01/2018', 'DD/MM/YYYY'),1,1,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (2,TO_DATE('01/01/2018', 'DD/MM/YYYY'),TO_DATE('01/01/2020', 'DD/MM/YYYY'),1,2,2);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (3,TO_DATE('01/05/2016', 'DD/MM/YYYY'),TO_DATE('01/05/2018', 'DD/MM/YYYY'),0,3,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (4,TO_DATE('15/12/2010', 'DD/MM/YYYY'),TO_DATE('15/12/2022', 'DD/MM/YYYY'),1,4,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (5,TO_DATE('01/01/2021', 'DD/MM/YYYY'),TO_DATE('01/01/2023', 'DD/MM/YYYY'),1,5,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (6,TO_DATE('03/09/2021', 'DD/MM/YYYY'),TO_DATE('03/09/2023', 'DD/MM/YYYY'),1,6,3);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (7,TO_DATE('02/03/2020', 'DD/MM/YYYY'),TO_DATE('02/03/2022', 'DD/MM/YYYY'),1,7,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (8,TO_DATE('03/09/2018', 'DD/MM/YYYY'),TO_DATE('03/09/2020', 'DD/MM/YYYY'),0,8,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (9,TO_DATE('01/06/2020', 'DD/MM/YYYY'),TO_DATE('01/06/2022', 'DD/MM/YYYY'),1,9,5);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (10,TO_DATE('03/01/2015', 'DD/MM/YYYY'),TO_DATE('03/01/2017', 'DD/MM/YYYY'),0,10,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (11,TO_DATE('03/09/2021', 'DD/MM/YYYY'),TO_DATE('03/09/2023', 'DD/MM/YYYY'),0,11,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (12,TO_DATE('03/01/2015', 'DD/MM/YYYY'),TO_DATE('03/01/2017', 'DD/MM/YYYY'),1,12,4);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (13,TO_DATE('02/01/2014', 'DD/MM/YYYY'),TO_DATE('02/01/2016', 'DD/MM/YYYY'),1,13,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (14,TO_DATE('10/06/2018', 'DD/MM/YYYY'),TO_DATE('10/06/2020', 'DD/MM/YYYY'),1,14,5);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (15,TO_DATE('10/06/2018', 'DD/MM/YYYY'),TO_DATE('10/06/2020', 'DD/MM/YYYY'),0,15,1);

INSERT INTO ppes (provision_id,receive,exchange,training,employee_id,ppe_id)
VALUES (16,TO_DATE('12/01/2018', 'DD/MM/YYYY'),TO_DATE('12/01/2020', 'DD/MM/YYYY'),1,1,1);

-- NaplnÏnÌ tabulky Training --

INSERT INTO training (training_id,training_name,frequency,last_update)
VALUES (1,'Safety induction training',2,CURRENT_DATE);

INSERT INTO training (training_id,training_name,frequency,last_update)
VALUES (2,'Fire safety induction training',2,CURRENT_DATE);

INSERT INTO training (training_id,training_name,frequency,last_update)
VALUES (3,'Forklift drivers training',1,CURRENT_DATE);

INSERT INTO training (training_id,training_name,frequency,last_update)
VALUES (4,'Working at height',1,CURRENT_DATE);

INSERT INTO training (training_id,training_name,frequency,last_update)
VALUES (5,'Training for electricians',3,CURRENT_DATE);


-- TRAINING_HISTORY --

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id  )
VALUES (1,TO_DATE('12/01/2016', 'DD/MM/YYYY'),TO_DATE('12/01/2016', 'DD/MM/YYYY'),'Ostrava',add_months('12/01/2016',24),1);

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (2,TO_DATE('01/01/2018', 'DD/MM/YYYY'),TO_DATE('01/01/2018', 'DD/MM/YYYY'),'P¯erov',add_months('01/01/2018',24),1);

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (3,TO_DATE('01/05/2016', 'DD/MM/YYYY'),TO_DATE('01/05/2016', 'DD/MM/YYYY'),'DolnÌ Beneöov',add_months('01/05/2016',24),1);

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (4,TO_DATE('15/12/2010','DD/MM/YYYY'),TO_DATE('15/12/2010','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('15/12/2010','DD/MM/YYYY'),24),1);

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (5,TO_DATE('01/01/2021','DD/MM/YYYY'),TO_DATE('01/01/2021','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('01/01/2021','DD/MM/YYYY'),24),1); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (6,TO_DATE('03/09/2021','DD/MM/YYYY'),TO_DATE('10/09/2021','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('03/09/2021','DD/MM/YYYY'),36),5); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (7,TO_DATE('02/03/2020','DD/MM/YYYY'),TO_DATE('02/03/2020','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('02/03/2020','DD/MM/YYYY'),24),1); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (8,TO_DATE('03/09/2018','DD/MM/YYYY'),TO_DATE('03/09/2018','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('03/09/2018','DD/MM/YYYY'),24),1); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (9,TO_DATE('01/06/2020','DD/MM/YYYY'),TO_DATE('01/06/2020','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('01/06/2020','DD/MM/YYYY'),24),1); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (10,TO_DATE('03/01/2015','DD/MM/YYYY'),TO_DATE('03/01/2015','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('03/01/2015','DD/MM/YYYY'),24),1); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (11,TO_DATE('03/09/2021','DD/MM/YYYY'),TO_DATE('03/09/2021','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('03/09/2021','DD/MM/YYYY'),24),1); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (12,TO_DATE('03/01/2015','DD/MM/YYYY'),TO_DATE('10/01/2015','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('03/01/2015','DD/MM/YYYY'),36),5); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (13,TO_DATE('02/01/2014','DD/MM/YYYY'),TO_DATE('02/01/2014','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('02/01/2014','DD/MM/YYYY'),24),1); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (14,TO_DATE('10/06/2018','DD/MM/YYYY'),TO_DATE('10/06/2018','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('10/06/2018','DD/MM/YYYY'),12),3); 

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (15,TO_DATE('10/06/2018','DD/MM/YYYY'),TO_DATE('10/06/2018','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('10/06/2018','DD/MM/YYYY'),24),1);

INSERT INTO training_history (employee_id ,training_start,training_end,place,expiration,training_id )
VALUES (1,TO_DATE('12/01/2016','DD/MM/YYYY'),TO_DATE('12/01/2016','DD/MM/YYYY'),'Ostrava',add_months(TO_DATE('10/06/2018','DD/MM/YYYY'),24),2);

-- ""CHECK"" --

INSERT INTO "CHECK"(check_id,check_name,frequency,last_update)
VALUES (1,'Entrance medical check',6,CURRENT_DATE);

INSERT INTO "CHECK"(check_id,check_name,frequency,last_update)
VALUES (2,'Medical check for welders',2,CURRENT_DATE);

INSERT INTO "CHECK"(check_id,check_name,frequency,last_update)
VALUES (3,'Medical check for forklift drivers',1,CURRENT_DATE);

INSERT INTO "CHECK"(check_id,check_name,frequency,last_update)
VALUES (4,'Medical check for electricians',3,CURRENT_DATE);

INSERT INTO "CHECK"(check_id,check_name,frequency,last_update)
VALUES (5,'Medical check for working at heights',1,CURRENT_DATE);

-- CHECKS --

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (1,TO_DATE('12/01/2016','DD/MM/YYYY'),TO_DATE('12/01/2016','DD/MM/YYYY'),add_months(TO_DATE('12/01/2016','DD/MM/YYYY'),72),1,1,TO_DATE ('12/01/2016','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (2,TO_DATE('01/01/2018','DD/MM/YYYY'),TO_DATE('01/01/2018','DD/MM/YYYY'),add_months(TO_DATE('01/01/2018','DD/MM/YYYY'),72),2,1,TO_DATE ('01/01/2018','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (3,TO_DATE('01/05/2016','DD/MM/YYYY'),TO_DATE('01/05/2016','DD/MM/YYYY'),add_months(TO_DATE('01/05/2016','DD/MM/YYYY'),72),3,1,TO_DATE ('20/08/2021','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (4,TO_DATE('15/12/2010','DD/MM/YYYY'),TO_DATE('15/12/2010','DD/MM/YYYY'),add_months(TO_DATE('15/12/2010','DD/MM/YYYY'),72),4,1,TO_DATE ('15/12/2010','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (5,TO_DATE('01/01/2021','DD/MM/YYYY'),TO_DATE('01/01/2021','DD/MM/YYYY'),add_months(TO_DATE('06/06/2018','DD/MM/YYYY'),72),5,1,TO_DATE ('01/01/2021','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (6,TO_DATE('03/09/2021','DD/MM/YYYY'),TO_DATE('03/09/2021','DD/MM/YYYY'),add_months(TO_DATE('03/09/2021','DD/MM/YYYY'),36),6,4,TO_DATE ('03/09/2021','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (7,TO_DATE('02/03/2020','DD/MM/YYYY'),TO_DATE('02/03/2020','DD/MM/YYYY'),add_months(TO_DATE('02/03/2020','DD/MM/YYYY'),72),7,1,TO_DATE ('02/03/2020','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (8,TO_DATE('03/09/2018','DD/MM/YYYY'),TO_DATE('03/09/2018','DD/MM/YYYY'),add_months(TO_DATE('03/09/2018','DD/MM/YYYY'),72),8,1,TO_DATE ('03/09/2018','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (9,TO_DATE('01/06/2020','DD/MM/YYYY'),TO_DATE('01/06/2020','DD/MM/YYYY'),add_months(TO_DATE('01/06/2020','DD/MM/YYYY'),72),9,1,TO_DATE ('01/06/2020','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (10,TO_DATE('03/01/2015','DD/MM/YYYY'),TO_DATE('03/01/2015','DD/MM/YYYY'),add_months(TO_DATE('03/01/2015','DD/MM/YYYY'),72),10,1,TO_DATE ('03/01/2015','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (11,TO_DATE('03/09/2021','DD/MM/YYYY'),TO_DATE('03/09/2021','DD/MM/YYYY'),add_months(TO_DATE('03/09/2021','DD/MM/YYYY'),72),11,1,TO_DATE ('03/09/2021','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (12,TO_DATE('03/01/2015','DD/MM/YYYY'),TO_DATE('03/01/2015','DD/MM/YYYY'),add_months(TO_DATE('03/01/2015','DD/MM/YYYY'),36),12,1,TO_DATE ('03/01/2015','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (13,TO_DATE('02/01/2014','DD/MM/YYYY'),TO_DATE('02/01/2014','DD/MM/YYYY'),add_months(TO_DATE('02/01/2014','DD/MM/YYYY'),72),13,1,TO_DATE ('02/01/2014','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (14,TO_DATE('10/06/2018','DD/MM/YYYY'),TO_DATE('10/06/2018','DD/MM/YYYY'),add_months(TO_DATE('10/06/2018','DD/MM/YYYY'),72),14,1,TO_DATE ('10/06/2018','DD/MM/YYYY'));

INSERT INTO checks(roch_id,check_date,check_end,expiration,employee_id,check_id,"Update")
VALUES (15,TO_DATE('10/06/2018','DD/MM/YYYY'),TO_DATE('10/06/2018','DD/MM/YYYY'),add_months(TO_DATE('10/06/2018','DD/MM/YYYY'),72),15,1,TO_DATE ('10/06/2018','DD/MM/YYYY'));

-- FACTOR --

INSERT INTO factor(factor_id ,factor_name,last_update)
VALUES(1,'Noise',CURRENT_DATE);

INSERT INTO factor(factor_id ,factor_name,last_update)
VALUES(2,'Vibration',CURRENT_DATE);

INSERT INTO factor(factor_id ,factor_name,last_update)
VALUES(3,'Chemical substances',CURRENT_DATE);

INSERT INTO factor(factor_id ,factor_name,last_update)
VALUES(4,'Dust',CURRENT_DATE);

INSERT INTO factor(factor_id ,factor_name,last_update)
VALUES(5,'Local muscle strain',CURRENT_DATE);


-- FACTORS --

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(1,TO_DATE('01/01/2021','DD/MM/YYYY'),3,add_months(TO_DATE('01/01/2021','DD/MM/YYYY'),36),TO_DATE('01/01/2021','DD/MM/YYYY'),1,5);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(2,TO_DATE('01/01/2021','DD/MM/YYYY'),0,NULL,TO_DATE('01/01/2021','DD/MM/YYYY'),2,1);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(3,TO_DATE('01/01/2021','DD/MM/YYYY'),0,NULL,TO_DATE('01/01/2021','DD/MM/YYYY'),3,1);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(4,TO_DATE('01/01/2021','DD/MM/YYYY'),0,NULL,TO_DATE('01/01/2021','DD/MM/YYYY'),4,1);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(5,TO_DATE('01/01/2021','DD/MM/YYYY'),0,NULL,TO_DATE('01/01/2021','DD/MM/YYYY'),5,1);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(6,TO_DATE('01/01/2021','DD/MM/YYYY'),3,add_months(TO_DATE('01/01/2021','DD/MM/YYYY'),36),TO_DATE('01/01/2021','DD/MM/YYYY'),6,3);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(7,TO_DATE('01/01/2021','DD/MM/YYYY'),0,NULL,TO_DATE('01/01/2021','DD/MM/YYYY'),7,1);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(8,TO_DATE('01/01/2021','DD/MM/YYYY'),0,NULL,TO_DATE('01/01/2021','DD/MM/YYYY'),8,1);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(9,TO_DATE('01/01/2021','DD/MM/YYYY'),3,add_months(TO_DATE('01/01/2021','DD/MM/YYYY'),36),TO_DATE('01/01/2021','DD/MM/YYYY'),9,3);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(10,TO_DATE('01/01/2021','DD/MM/YYYY'),0,NULL,TO_DATE('01/01/2021','DD/MM/YYYY'),10,1);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(11,TO_DATE('01/01/2021','DD/MM/YYYY'),0,NULL,TO_DATE('01/01/2021','DD/MM/YYYY'),11,1);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(12,TO_DATE('01/01/2021','DD/MM/YYYY'),0,add_months(TO_DATE('01/01/2021','DD/MM/YYYY'),36),TO_DATE('01/01/2021','DD/MM/YYYY'),12,3);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(13,TO_DATE('01/01/2021','DD/MM/YYYY'),0,add_months(TO_DATE('01/01/2021','DD/MM/YYYY'),36),TO_DATE('01/01/2021','DD/MM/YYYY'),13,5);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(14,TO_DATE('01/01/2021','DD/MM/YYYY'),0,add_months(TO_DATE('01/01/2021','DD/MM/YYYY'),36),TO_DATE('01/01/2021','DD/MM/YYYY'),14,5);

INSERT INTO factors(rof,last_measurement,frequency,nextmeasurement,record_date,employee_id,factor_id)
VALUES(15,TO_DATE('01/01/2021','DD/MM/YYYY'),0,NULL,TO_DATE('01/01/2021','DD/MM/YYYY'),15,1);

-- Pokus --

ALTER session set nls_date_format='DD-MM-YYYY HH24:MI'; 

-- whours --

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),2,TO_DATE('2021/08/22 6:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),1);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),24.5,TO_DATE('2021/08/22 15:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),2);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 19:00:00', 'yyyy/mm/dd hh24:mi:ss'),3,TO_DATE('2021/08/22 6:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 19:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),3);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),10,TO_DATE('2021/08/22 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),4);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),15,TO_DATE('2021/08/22 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),5);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),0,TO_DATE('2021/08/22 4:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 06:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),6);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),24.5,TO_DATE('2021/08/22 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),7);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 15:00:00', 'yyyy/mm/dd hh24:mi:ss'),20,TO_DATE('2021/08/22 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 15:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),8);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 17:30:00', 'yyyy/mm/dd hh24:mi:ss'),5,TO_DATE('2021/08/22 4:45:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 17:30:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 06:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),9);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 17:00:00', 'yyyy/mm/dd hh24:mi:ss'),3.5,TO_DATE('2021/08/22 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 17:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),10);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),0,TO_DATE('2021/08/22 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),11);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),10,TO_DATE('2021/08/22 6:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 06:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),12);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),10.5,TO_DATE('2021/08/22 6:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 06:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),13);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),11,TO_DATE('2021/08/22 6:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 06:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),14);

INSERT INTO whours(record_date,holiday,arrival,leaving,shiftb,shifte,employee_id)
VALUES (TO_DATE('2021/08/22 15:00:00', 'yyyy/mm/dd hh24:mi:ss'),25,TO_DATE('2021/08/22 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 15:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/08/22 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/08/22 16:00:00', 'yyyy/mm/dd hh24:mi:ss'),15);

-- ACCOUNT --

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (1,1234567890,0600,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (2,101656009,2250,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (3,8460480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (6,9460480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (7,9360480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (8,9260480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (9,7460480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (10,6460480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (11,5460480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (12,4460480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (13,2460480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (14,1460480583,0800,CURRENT_DATE);

INSERT INTO ACCOUNT(account_id,a_number,bank_code,last_update)
VALUES (15,0460480583,0800,CURRENT_DATE);

-- METHODS --

INSERT INTO METHODS(m_id,mail,cash,last_update)
VALUES(4,1,NULL,CURRENT_DATE);

INSERT INTO METHODS(m_id,mail,cash,last_update)
VALUES(5,NULL,1,CURRENT_DATE);


-- PAYMENTS --

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (1,1,5000,1,NULL,1,Current_date);

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (2,1,2500,2,NULL,2,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (3,3,15000,3,NULL,3,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (4,3,10000,4,4,NULL,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (5,1,3000,5,5,NULL,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (6,1,4000,6,NULL,6,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (7,1,500,7,NULL,7,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (8,1,1000,8,NULL,8,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (9,1,5000,9,NULL,9,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (10,3,150000,10,NULL,10,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (11,1,5000,11,NULL,11,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (12,1,6000,12,NULL,12,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (13,1,3000,13,NULL,13,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (14,1,3000,14,NULL,14,Current_date); 

INSERT INTO payments(p_id,category,bonus,employee_id,m_id,account_id,last_update)
VALUES (15,1,-5000,15,NULL,15,Current_date); 

-- SALARY CHANGE --

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),50000,null,'Ellen Kacalova',1);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),25000,null,'Michal Kraninger',2);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),150000,null,'Michal Kraninger',3);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),100000,80000,'Michal Kraninger',4);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),30000,29000,'Michal Kraninger',5);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),40000,31000,'Michal Kraninger',6);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),25000,20000,'Michal Kraninger',7);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),40000,30000,'Michal Kraninger',8);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),50000,NULL,'Michal Kraninger',9);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),150000,125000,'Michal Kraninger',10);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),51000,25000,'Michal Kraninger',11);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),60000,59000,'Michal Kraninger',12);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),30000,NULL,'Michal Kraninger',13);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),30000,NULL,'Michal Kraninger',14);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/08/22 04:45:00', 'yyyy/mm/dd hh24:mi:ss'),70000,NULL,'Michal Kraninger',15);

INSERT INTO salary_change(date_modified,salary,old_salary,modified_by ,employee_id)
VALUES (TO_DATE('2021/11/22 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),50000,50000,'Ellen Kacalova',1);

--  sickness --

INSERT INTO sickness(sick_id,sick,sicknesss,sicknesse,hospital,hospitals,hospitale,noosds,record_date,employee_id)
VALUES(1,1,TO_DATE('2021/07/03 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/07/06 08:00:00', 'yyyy/mm/dd hh24:mi:ss'),0,NULL,
NULL,3,CURRENT_DATE,1);

INSERT INTO sickness(sick_id,sick,sicknesss,sicknesse,hospital,hospitals,hospitale,noosds,record_date,employee_id)
VALUES(4,1,TO_DATE('2021/05/05 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),TO_DATE('2021/05/10 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),1,TO_DATE('2021/05/05 14:00:00', 'yyyy/mm/dd hh24:mi:ss'),
TO_DATE('2021/05/10 8:00:00', 'yyyy/mm/dd hh24:mi:ss'),5,CURRENT_DATE,4);

INSERT INTO sickness(sick_id,sick,sicknesss,sicknesse,hospital,hospitals,hospitale,noosds,record_date,employee_id)
VALUES(5,0,NULL,NULL,0,NULL,NULL,0,CURRENT_DATE,5);


-- disclass --

INSERT INTO disclass (classod,description,last_update)
VALUES (0,'No degree of disability',CURRENT_DATE);

INSERT INTO disclass (classod,description,last_update)
VALUES (1,'1st degree of disability',CURRENT_DATE);

INSERT INTO disclass (classod,description,last_update)
VALUES (2,'2st degree of disability',CURRENT_DATE);

INSERT INTO disclass (classod,description,last_update)
VALUES (3,'3rd degree of disability',CURRENT_DATE);

-- dl --

INSERT INTO dl (dl_choices,description,last_update)
VALUES(0,'No driving license',CURRENT_DATE);

INSERT INTO dl (dl_choices,description,last_update)
VALUES(1,'motorcycle',CURRENT_DATE);

INSERT INTO dl (dl_choices,description,last_update)
VALUES(2,'car',CURRENT_DATE);

INSERT INTO dl (dl_choices,description,last_update)
VALUES(3,'Lorry',CURRENT_DATE);

INSERT INTO dl (dl_choices,description,last_update)
VALUES(4,'bus',CURRENT_DATE);

-- education --

INSERT INTO education(education_id,description,last_update)
VALUES(0,'No formal education',CURRENT_DATE);

INSERT INTO education(education_id,description,last_update)
VALUES(1,'Primary school',CURRENT_DATE);

INSERT INTO education(education_id,description,last_update)
VALUES(2,'Secondary industrial school',CURRENT_DATE);

INSERT INTO education(education_id,description,last_update)
VALUES(3,'Grammar school',CURRENT_DATE);

INSERT INTO education(education_id,description,last_update)
VALUES(4,'Business academy',CURRENT_DATE);

INSERT INTO education(education_id,description,last_update)
VALUES(5,'Vocational school',CURRENT_DATE);

INSERT INTO education(education_id,description,last_update)
VALUES(6,'University',CURRENT_DATE);

-- marital_status --

INSERT INTO marital_status(marital_id,description,last_update)
VALUES(1,'Single',CURRENT_DATE);

INSERT INTO marital_status(marital_id,description,last_update)
VALUES(2,'Married',CURRENT_DATE);

INSERT INTO marital_status(marital_id,description,last_update)
VALUES(3,'Widow/Widower',CURRENT_DATE);

INSERT INTO marital_status(marital_id,description,last_update)
VALUES(4,'Divorced',CURRENT_DATE);

-- pi --

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (1,123336,151525,1,utl_raw.cast_to_raw('This is a blob description'),0,0,NULL,CURRENT_DATE,1,0,3,6,1);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (2,123336,151525,1,utl_raw.cast_to_raw('Pokus'),2,0,'Sex',CURRENT_DATE,2,0,2,6,1);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (3,123336,151525,0,NULL,0,0,'Reading',CURRENT_DATE,3,0,3,6,0);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (4,123336,151525,1,utl_raw.cast_to_raw('Zamestnanec cislo 4'),1,0,'Doing nothing',CURRENT_DATE,4,0,4,6,4);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (5,123336,151525,0,utl_raw.cast_to_raw('Zamestnanec cislo 5'),0,0,' ',CURRENT_DATE,5,0,1,6,3);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (6,323333,111666,0,utl_raw.cast_to_raw('Zamestnanec cislo 6'),0,1,'DB',CURRENT_DATE,6,2,1,1,4);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (7,974617,0571058,0,utl_raw.cast_to_raw('Zamestnanec cislo 7'),0,0,'Swimming ',CURRENT_DATE,7,0,1,6,3);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (8,1139210,5105296,0,utl_raw.cast_to_raw('Zamestnanec cislo 8'),3,0,'Skettles',CURRENT_DATE,8,0,4,5,3);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (9,123336,151525,1,utl_raw.cast_to_raw('Zamestnanec cislo 9'),0,0,'Football',CURRENT_DATE,9,0,3,4,3);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (10,749888,687766,0,utl_raw.cast_to_raw('Zamestnanec cislo 10'),1,0,'Ice Hockey ',CURRENT_DATE,10,0,2,3,3);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (11,690421,684517,0,utl_raw.cast_to_raw('Zamestnanec cislo 11'),0,1,'SEX ',CURRENT_DATE,11,1,1,2,3);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (12,484929,478092,1,utl_raw.cast_to_raw('Zamestnanec cislo 12'),2,0,'Money ',CURRENT_DATE,12,0,4,1,3);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (13,901656,096139,0,utl_raw.cast_to_raw('Zamestnanec cislo 13'),0,0,'Alcohol ',CURRENT_DATE,13,0,3,3,3);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (14,446397,096739,0,utl_raw.cast_to_raw('Zamestnanec cislo 14'),2,0,'Drugs ',CURRENT_DATE,14,0,2,2,3);

INSERT INTO pi (pi_id, id_number, dl_number, marital_status, photo, noochildren, disability, hobbies, record_date, employee_id, classod, marital_id, education_id, dl_choices)
VALUES (15,856080,402665,1,utl_raw.cast_to_raw('Zamestnanec cislo 15'),0,1,' ',CURRENT_DATE,15,1,1,6,2);



