----------------------------------
---- Postgresql Query Tools ------
----------------------------------
---- DDL : Table Create Query ----
----------------------------------




--- make it a "begin" format so it can save the proggres
begin;

CREATE TABLE table_m3(
	AGE integer,
	WORKclass varchar(50),
	fnlWGT integer,
	EDUCATION varchar(30),
	"educational-num" integer,
	"marital-status" varchar(30),
	occupation varchar(50),
	RELATIONSHIP varchar(50),
	RACE varchar(50),
	GENDER varchar(20),
	"capital-gain" integer,
	"capital-loss" integer,
	"hours-per-week" integer,
	"native-country" varchar(20) 
);
--- commit to safe current proggress
commit;
--- use a rollback if there is a mistake on making table
ROLLBACK;
--- drop table
DROP TABLE table_m3;
--- make a copy for a table m3 
copy table_m3
from 'C:\Program Files\PostgreSQL\16\P2M3_adhy_arya_data_raw.csv'
delimiter ','
csv header ;
--- show table m3
select * from table_m3