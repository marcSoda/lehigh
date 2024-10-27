drop index idx_location;
drop index idx_charge;
drop index idx_rental;
drop index idx_reservation;
drop index idx_vehicle;
drop table charge;
drop table reservation;
drop table rental;
drop table vehicle;
drop table location;
drop table customer;
drop table membership;
drop table discount;
drop table license;

create table license(
    license_id integer GENERATED ALWAYS AS IDENTITY,
    license_number varchar(256) not null,
    state varchar(64) not null,
    age integer not null,
    primary key(license_id),
    CONSTRAINT dln_unique UNIQUE (license_number)
);

create table discount(
    discount_id integer GENERATED ALWAYS AS IDENTITY,
    percent_off numeric(3) not null,
    description varchar(512) not null,
    primary key(discount_id)
);

create table membership(
    membership_id integer GENERATED ALWAYS AS IDENTITY,
    group_name varchar(256) not null,
    discount_id integer not null,
    foreign key(discount_id) references discount(discount_id),
    primary key(membership_id),
    CONSTRAINT mid_unique UNIQUE (group_name)
);

create table customer (
    customer_id integer GENERATED ALWAYS AS IDENTITY,
    address varchar(256) not null,
    name varchar(256) not null,
    license_id integer not null,
    membership_id integer not null,
    foreign key(license_id) references license(license_id),
    foreign key(membership_id) references membership(membership_id),
    primary key(customer_id)
);

create table location(
    location_id integer GENERATED ALWAYS AS IDENTITY,
    street varchar(128) not null,
    city varchar(128) not null,
    state varchar(64) not null,
    zip integer not null,
    primary key(location_id)
);
CREATE INDEX idx_location ON location (city);

create table vehicle (
    vehicle_id integer GENERATED ALWAYS AS IDENTITY,
    make varchar(64) not null,
    model varchar(64) not null,
    year number(4) not null,
    color varchar(16) not null,
    type varchar(16) not null,
    vin varchar(17) not null,
    odo integer not null,
    location_id integer not null,
    foreign key(location_id) references location(location_id),
    primary key(vehicle_id)
);
CREATE INDEX idx_vehicle ON vehicle (location_id);

create table rental(
    rental_id integer GENERATED ALWAYS AS IDENTITY,
    taken date not null,
    returned date,
    rate numeric(10,4) not null,
    customer_id integer not null,
    vehicle_id integer not null,
    foreign key(customer_id) references customer(customer_id),
    foreign key(vehicle_id) references vehicle(vehicle_id),
    primary key(rental_id)
);
CREATE INDEX idx_rental ON rental (customer_id, vehicle_id, returned);

create table reservation (
    reservation_id integer GENERATED ALWAYS AS IDENTITY,
    created date not null,
    customer_id integer not null,
    vehicle_id integer not null,
    foreign key(customer_id) references customer(customer_id),
    foreign key(vehicle_id) references vehicle(vehicle_id),
    primary key(reservation_id)
);
CREATE INDEX idx_reservation ON reservation (customer_id, vehicle_id, created);

create table charge(
    charge_id integer GENERATED ALWAYS AS IDENTITY,
    fuel numeric(10,4) not null,
    dropoff numeric(10,4) not null,
    insurance numeric(10,4) not null,
    other numeric(10,4) not null,
    complete number(1) default 0 not null,
    rental_id integer not null,
    foreign key(rental_id) references rental(rental_id),
    primary key(charge_id)
);
CREATE INDEX idx_charge ON charge (rental_id, complete);

INSERT INTO location VALUES (DEFAULT,'Blackheath Lane','Fayetteville','Virginia','68254');
INSERT INTO location VALUES (DEFAULT,'Cavell Vale','Las Vegas','Oklahoma','55672');
INSERT INTO location VALUES (DEFAULT,'Blackheath Tunnel','London','Tennessee','63955');
INSERT INTO location VALUES (DEFAULT,'Beaconsfield Tunnel','Henderson','Vermont','95558');
INSERT INTO location VALUES (DEFAULT,'Bury Tunnel','Stockton','Nevada','12998');
INSERT INTO location VALUES (DEFAULT,'Kimberley Avenue','Tokyo','Maine','18854');
INSERT INTO location VALUES (DEFAULT,'Church Tunnel','Seattle','Iowa','25829');
INSERT INTO location VALUES (DEFAULT,'Chancellor Pass','Richmond','Missouri','20365');
INSERT INTO location VALUES (DEFAULT,'Geffrye Pass','Baltimore','New Jersey','41073');
INSERT INTO location VALUES (DEFAULT,'Birkenhead Lane','Moreno Valley','Vermont','62480');
INSERT INTO location VALUES (DEFAULT,'Gate Vale','Fayetteville','Maryland','43004');
INSERT INTO location VALUES (DEFAULT,'Yorkshire Walk','Laredo','Utah','77297');
INSERT INTO location VALUES (DEFAULT,'Charterhouse Alley','Bridgeport','Mississippi','59033');
INSERT INTO location VALUES (DEFAULT,'Falconberg Lane','Bakersfield','New Mexico','80074');
INSERT INTO location VALUES (DEFAULT,'Blean Alley','Lisbon','Iowa','70224');
INSERT INTO location VALUES (DEFAULT,'Liverpool Grove','Springfield','Michigan','56224');
INSERT INTO location VALUES (DEFAULT,'King Pass','Dallas','Kentucky','77492');
INSERT INTO location VALUES (DEFAULT,'Bacton Vale','Columbus','Nevada','11034');
INSERT INTO location VALUES (DEFAULT,'Cloth Hill','Amarillo','Pennsylvania','59680');
INSERT INTO location VALUES (DEFAULT,'Addison Alley','Bucharest','California','51087');

INSERT INTO vehicle VALUES (DEFAULT,'Mazda','CX5','2015','Ruby','truck','fiiLRbNzRLM0jR3hN','130756','6');
INSERT INTO vehicle VALUES (DEFAULT,'Dacia','Logan','1961','Dark red','truck','vNHquoQ1CP5pRfS22','244720','17');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M6','1979','Magenta','van','pKejypP7XfOFIhh69','262634','1');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','X1','2014','Ruby','coup','IWESfDUGDFdvSOkps','450121','1');
INSERT INTO vehicle VALUES (DEFAULT,'Dacia','Logan','2010','Lavender','van','TL1DvtaxtpTLo2b38','46120','8');
INSERT INTO vehicle VALUES (DEFAULT,'Fiat','Argo','1971','Maroon','van','JCunBYYY7L2dDRWD8','277503','10');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i30','1990','Turquoise','coup','yofmJm3aumrpJrR4w','346396','3');
INSERT INTO vehicle VALUES (DEFAULT,'Mazda','3','1986','Fuchsia','sedan','NIsJyEAMdNt3wxi9D','152342','6');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i20','1984','Cerise','sedan','rXU8u30Zm5eKQ5hqd','339766','4');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i40','1961','Sky blue','van','WmIalZRbxHrqNtT3d','78365','19');
INSERT INTO vehicle VALUES (DEFAULT,'Ford','Focus','2011','Fuchsia','truck','4IVuVqeLYXQr9AgMt','240285','1');
INSERT INTO vehicle VALUES (DEFAULT,'Mitsubishi','Outlander','2022','Gold','van','h0BEvkPAbrQ6qbbNe','407143','4');
INSERT INTO vehicle VALUES (DEFAULT,'Mitsubishi','Lancer','1972','Barbie Pink','truck','FSjS9oLKGxzcGnxBq','274088','10');
INSERT INTO vehicle VALUES (DEFAULT,'Fiat','Panda','1976','Lime','truck','eblIbDrqNqRpscCFv','292999','13');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','V8','2007','Brown','truck','e5zEkCQMqeMkBotHw','445817','10');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','R8','1982','Carmine','sedan','ecXv1GQCH6H1lmnBc','18839','19');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','R8','1991','Capri','coup','8eNDog0j3Gk3byY6h','454217','14');
INSERT INTO vehicle VALUES (DEFAULT,'Mazda','3','1976','coral','truck','36IaqugcR3lC9caRS','360643','20');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','X3','1967','Purple','sedan','7zNTNxO3jvkrIMWDa','79690','1');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M2','2015','Maroon','coup','7buRW2sjgs4SJEGXJ','207391','8');
INSERT INTO vehicle VALUES (DEFAULT,'Mazda','3','1996','Fuchsia','coup','r1SwAZY69QVemN21I','302965','13');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i40','2007','Green','coup','tIubtw6kSpBZwgSgW','78287','7');
INSERT INTO vehicle VALUES (DEFAULT,'Mitsubishi','Outlander','2015','Beige','sedan','biBsNZvdWQRnfPdZ4','162716','13');
INSERT INTO vehicle VALUES (DEFAULT,'Citroen','Nemo','1988','Camel','van','YIREgN66cSO41EPQ5','211005','5');
INSERT INTO vehicle VALUES (DEFAULT,'Ford','Mustang','2015','Lavender','truck','8Mn7gGdzwzRdZ0DVr','19512','5');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','R8','1990','Lime','truck','g5zKChN7shAS96MuE','41156','12');
INSERT INTO vehicle VALUES (DEFAULT,'Ford','Mustang','2009','Champagne','van','Ly6F166UVAKoptSDf','349616','17');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M5','1962','coral','van','Z7AUyAVne1sR21lD7','317931','7');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M2','1966','Camel','sedan','QyRJndsKkb96AgAzB','92181','2');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q6','2008','Fuchsia','truck','FEKuXAlDZiLEd1Y9G','142876','9');
INSERT INTO vehicle VALUES (DEFAULT,'Ford','Fiesta','1967','Carmine','sedan','FNdXJn8jJhYiaMaX3','303123','13');
INSERT INTO vehicle VALUES (DEFAULT,'Opel','Insignia','2007','Sky blue','van','pVaF2I8OiSwIqP4Me','162486','16');
INSERT INTO vehicle VALUES (DEFAULT,'Fiat','Panda','1987','Orange','coup','OVHGah39xeRieRZaT','437709','8');
INSERT INTO vehicle VALUES (DEFAULT,'Mazda','CX5','1974','Magenta','truck','QNBdJ9Qb0ox0IHu3I','443712','17');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q7','2000','Sky blue','sedan','HxKVAB9DdKU2gneAH','320483','9');
INSERT INTO vehicle VALUES (DEFAULT,'Fiat','Tipo','1995','Auburn','van','FTka8QKH3CLGF0nxx','325495','6');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','R8','1961','Magenta','van','gWaRA5CBl4rYmf3Ku','312814','19');
INSERT INTO vehicle VALUES (DEFAULT,'Ford','Kuga','2017','Dark red','coup','vhxtsdEUxb3t02GJE','248710','12');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Cabriolet','1985','Red','coup','Q9n7sfjbcOYPS2amE','414413','11');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','A5','1986','Azure','coup','0g1DZlYtdcdaWF6US','189495','4');
INSERT INTO vehicle VALUES (DEFAULT,'Kia','Soul','2015','Maroon','van','UUFDzVPIx28sYCnAg','409782','16');
INSERT INTO vehicle VALUES (DEFAULT,'Ford','Fiesta','1971','Champagne','sedan','HhGxQD3VtFQSpoGOS','279348','4');
INSERT INTO vehicle VALUES (DEFAULT,'Fiat','500','1999','Dark Red','coup','W7NiImk8YryJVhyEW','328727','13');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','R8','2004','Rosegold','sedan','3JhICNMGF3GqwlcpT','137109','5');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Cabriolet','1965','Gray','coup','PPWn9gw40OdD0WelD','123423','13');
INSERT INTO vehicle VALUES (DEFAULT,'Kia','Soul','1983','Dark Red','sedan','XjO9n6HJOw7zcGXTw','467628','16');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M3','1998','Dark Red','van','CrXpgpM15l6i4NRV4','311548','19');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i30','2008','Green','van','koqgF4owSlPtdrGu8','265727','8');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i40','2007','Magenta','sedan','r0V45biIQ5BfU0rRw','306132','13');
INSERT INTO vehicle VALUES (DEFAULT,'Fiat','Tipo','1998','Olive','truck','wDoyot5l3XET1WJql','106373','10');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q7','1975','Azure','sedan','8QGCBQ52IJY4TlMG4','219219','9');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i35','2020','Cerise','van','9NJXhIW0pqczj1GIb','358283','17');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M6','2008','Fuchsia','truck','UMNKLCAnpCDvflABD','323515','9');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M2','1977','coral','truck','hNBskrmkYM6JpdocL','450525','7');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','Elantra','1983','Rosegold','truck','soqzsAn0qoz94FVaK','55468','7');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q5','2014','Black','coup','FatSGlGyA2C5vpMST','494308','3');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','Elantra','1989','Magenta','coup','MzsJ0UeZhnj3hjDtK','450829','1');
INSERT INTO vehicle VALUES (DEFAULT,'Fiat','500','1991','Olive','sedan','iYVcyxmPp4wXoJhig','110389','7');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','A6','1985','Aqua','truck','SXlMHwYfXR5I7tz7o','436573','10');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i20','1981','Auburn','truck','8LYkWx8jWfZADhHY2','137509','4');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q6','1985','Champagne','sedan','u3uetwn6cJl95Bd6j','292403','14');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M6','2019','Purple','van','dO1DUYrL5l4OEmtwD','298323','13');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q5','1984','Gold','truck','09QwrJiMlQhw3Hv3x','3715','6');
INSERT INTO vehicle VALUES (DEFAULT,'Ford','Fiesta','2004','Magenta','van','3A6Beg9dYeZbYNcVB','241070','20');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M3','1960','Magenta','sedan','k6zz6VCvJyImLGtiS','246791','20');
INSERT INTO vehicle VALUES (DEFAULT,'Kia','Soul','1988','Rosewood','sedan','qAkTnsgTX3kJnbIXq','125036','8');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q5','2005','Magenta','truck','rq0RLMI2lQVbcgGmT','401858','6');
INSERT INTO vehicle VALUES (DEFAULT,'Fiat','500','1985','Beige','sedan','vJZoe0m7q6BqBMrcQ','16970','13');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M4','2018','Dark Red','sedan','ZCLeNCnVSZCtOHZsd','385187','13');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i30','2017','Dark Red','sedan','Im1rChZH6Eybdc3YV','366669','8');
INSERT INTO vehicle VALUES (DEFAULT,'Mitsubishi','Outlander','2005','Peach','truck','ccJMwHbqyFZwnP1mm','142127','20');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','A5','1976','Dark Red','van','jixz8FsFmkFQ4K6Wu','331364','18');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M3','1998','Cyan','sedan','Gu0KycIdgMjo1zArr','7206','16');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i20','1982','Orange','coup','gV6ME1E0x1U15yI6w','193813','5');
INSERT INTO vehicle VALUES (DEFAULT,'Opel','Insignia','2005','Mauve','van','4qyKpYou5SGMRcUex','37224','3');
INSERT INTO vehicle VALUES (DEFAULT,'Opel','Insignia','2012','Ruby','truck','jTItgmgkYJKQpHcqH','499009','16');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','X3','2007','Rust','truck','5y0r7IeQVFFfAZpkm','438821','16');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M5','1974','Beige','van','swpgEEXe1kyp2bAq9','160221','17');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','X1','1971','Emerald','coup','Innenx1VsPWzuRBrB','78634','11');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','SantaFe','2018','Capri','coup','Ox5HxmLxF6au9Nx7T','82271','4');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','M4','2013','Champagne','van','J6wBfYLVBZ95N5nLB','327954','5');
INSERT INTO vehicle VALUES (DEFAULT,'Mitsubishi','Lancer','2020','Purple','truck','ljPom9v5bnpYwkhOY','488050','10');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','SantaFe','1961','Aquamarine','coup','zkqYsIAvH3z6rYZOA','385826','10');
INSERT INTO vehicle VALUES (DEFAULT,'Kia','Sportage','2009','Lavender','van','iCHN2fUrLxYEgtepn','330817','8');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q3','1978','Amethyst','truck','BU7kpkg2CKF2iPYfP','254744','7');
INSERT INTO vehicle VALUES (DEFAULT,'Ford','Kuga','1969','Black','coup','JjYGD0dto0KWGgM1L','245600','2');
INSERT INTO vehicle VALUES (DEFAULT,'Hyundai','i30','2009','Carmine','coup','DL7WZiRKSgAIw8TmI','449959','15');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q5','1968','Azure','van','78dXfEKG4AofWPS2M','409154','18');
INSERT INTO vehicle VALUES (DEFAULT,'Dacia','Logan','2005','Mauve','sedan','oLADD4MiNbjywwb0B','433639','16');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','R8','1967','Lavender','sedan','6T9NaicMl03KD1Fc7','77135','1');
INSERT INTO vehicle VALUES (DEFAULT,'Mitsubishi','Outlander','1964','Blue','truck','rhhmp2g3sGlNUw3Gl','92131','1');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Cabriolet','2002','Carmine','coup','Jm1XQhRa2sA4k68WQ','168169','19');
INSERT INTO vehicle VALUES (DEFAULT,'Mazda','6','1975','Magenta','sedan','cNFAoshpafQpgbjTw','219974','6');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','A5','1971','Carmine','coup','aoQX6gP6uyTLaOAcP','84589','16');
INSERT INTO vehicle VALUES (DEFAULT,'Citroen','C3','2003','Turquoise','sedan','cLSBab8HyOEQSd8As','385346','3');
INSERT INTO vehicle VALUES (DEFAULT,'Fiat','Tipo','1967','Emerald','coup','60yQequFMdU65UWUE','308239','15');
INSERT INTO vehicle VALUES (DEFAULT,'Audi','Q1','1972','Dark Red','sedan','87Q5DH8PlelJ1UiUL','47261','17');
INSERT INTO vehicle VALUES (DEFAULT,'Dacia','Logan','1963','Gray','van','DtNHOAJWigsdJJWWr','160767','18');
INSERT INTO vehicle VALUES (DEFAULT,'BMW','X5','1971','Lime','sedan','A0qxnKgAA9YvHXdbx','154496','20');
INSERT INTO vehicle VALUES (DEFAULT,'Mitsubishi','Outlander','2012','Black','truck','Wdlft1bvfNJdi5VMd','72264','12');

INSERT INTO license VALUES (DEFAULT,'X7DlQr3z3pRu4QP','Oklahoma','93');
INSERT INTO license VALUES (DEFAULT,'wglDoH3aQrZOZHh','Arkansas','35');
INSERT INTO license VALUES (DEFAULT,'oJTmSxgqaQROdiC','Oregon','31');
INSERT INTO license VALUES (DEFAULT,'JYEes0JLAAdkdhu','South Dakota','82');
INSERT INTO license VALUES (DEFAULT,'lFjSesFWO1mLwIW','Kansas','49');
INSERT INTO license VALUES (DEFAULT,'8IBn5TKbRxCG6sg','Idaho','77');
INSERT INTO license VALUES (DEFAULT,'67dwpp8He8g9O1G','Illinois','35');
INSERT INTO license VALUES (DEFAULT,'jlnHqvZ1M5CQRJF','West Virginia','29');
INSERT INTO license VALUES (DEFAULT,'MZshTdCgbUnbC66','South Carolina','84');
INSERT INTO license VALUES (DEFAULT,'J46kQLE5cu60Mza','Illinois','37');
INSERT INTO license VALUES (DEFAULT,'oqqlPIllOavZJ8F','Idaho','60');
INSERT INTO license VALUES (DEFAULT,'ALpd6Nnl1sFpPfM','Oregon','89');
INSERT INTO license VALUES (DEFAULT,'RQAM3wV8CuTgSv1','Arizona','78');
INSERT INTO license VALUES (DEFAULT,'zPtqsdC9Yaf595j','Pennsylvania','46');
INSERT INTO license VALUES (DEFAULT,'rziRGYUe3cUFi6O','Illinois','63');
INSERT INTO license VALUES (DEFAULT,'q6ULiw8MfV0DDzt','Wyoming','25');
INSERT INTO license VALUES (DEFAULT,'EBsbs0YLyAs4zdQ','Georgia','58');
INSERT INTO license VALUES (DEFAULT,'PmBGJCkEkDdb0nk','Nevada','65');
INSERT INTO license VALUES (DEFAULT,'2NpE7F59tvPSq2j','Georgia','68');
INSERT INTO license VALUES (DEFAULT,'jHI458IXIx5u8XH','Rhode Island','46');
INSERT INTO license VALUES (DEFAULT,'lD1XSgGHZRUv4K4','Hawaii','44');
INSERT INTO license VALUES (DEFAULT,'Ddvn6OhryRAPvXI','Iowa','54');
INSERT INTO license VALUES (DEFAULT,'QgpNRkCwfrylNmA','Nebraska','77');
INSERT INTO license VALUES (DEFAULT,'UoOoozTqdOG2n3L','Florida','63');
INSERT INTO license VALUES (DEFAULT,'4XQDw2t0bgtFlp9','Indiana','29');
INSERT INTO license VALUES (DEFAULT,'DeEulrVgxW641ym','Minnesota','77');
INSERT INTO license VALUES (DEFAULT,'sniohvexNSkk2jG','New Mexico','68');
INSERT INTO license VALUES (DEFAULT,'zrZvSq4g9gx6jrG','Massachusetts','90');
INSERT INTO license VALUES (DEFAULT,'6W1Ygsj6YYu64wz','New Mexico','45');
INSERT INTO license VALUES (DEFAULT,'85s6Z6oIiPr1old','Tennessee','98');
INSERT INTO license VALUES (DEFAULT,'nukLT80y4SJkEt8','Florida','46');
INSERT INTO license VALUES (DEFAULT,'tLO6gZMQ7d2FtxQ','Maine','93');
INSERT INTO license VALUES (DEFAULT,'JE1Fp62Hfoeieft','Illinois','52');
INSERT INTO license VALUES (DEFAULT,'1SYCGUY5We8tKe2','Iowa','35');
INSERT INTO license VALUES (DEFAULT,'k3mLDOWyBvssd7J','South Carolina','52');
INSERT INTO license VALUES (DEFAULT,'5842DtdKSb4YUJl','Arizona','50');
INSERT INTO license VALUES (DEFAULT,'5zvLxDBNEjpN2hz','Connecticut','84');
INSERT INTO license VALUES (DEFAULT,'j4XPgJIBvwcaqlG','New Mexico','34');
INSERT INTO license VALUES (DEFAULT,'CT6LH42p8Jx7a4F','New Hampshire','64');
INSERT INTO license VALUES (DEFAULT,'kr3FYpmJCRjxX8i','Maryland','81');
INSERT INTO license VALUES (DEFAULT,'efFlcAeki9VDU1X','Oregon','37');
INSERT INTO license VALUES (DEFAULT,'AZiCnnTBEnS1kLO','Kansas','73');
INSERT INTO license VALUES (DEFAULT,'ajR4IqYe0qRGrZM','Texas','41');
INSERT INTO license VALUES (DEFAULT,'7JNt01vkXXMt7yG','Alabama','75');
INSERT INTO license VALUES (DEFAULT,'bxvP2sJhx7B5WNG','Missouri','52');
INSERT INTO license VALUES (DEFAULT,'NF85jrfUuTMv21L','Washington','27');
INSERT INTO license VALUES (DEFAULT,'7Y88FaDLD4uoBJF','West Virginia','68');
INSERT INTO license VALUES (DEFAULT,'lA2u2GP4HZ6v5G0','New Jersey','49');
INSERT INTO license VALUES (DEFAULT,'PrIYHp4ugxvqtKG','California','46');
INSERT INTO license VALUES (DEFAULT,'QLVRFStE7BHaIy4','Massachusetts','75');
INSERT INTO license VALUES (DEFAULT,'3XdLdV89WSdNYO0','West Virginia','71');
INSERT INTO license VALUES (DEFAULT,'zpVCKZYUKnxUY91','Kentucky','28');
INSERT INTO license VALUES (DEFAULT,'QtqGxzXSt9Q0jwR','Mississippi','62');
INSERT INTO license VALUES (DEFAULT,'8Pmot8E1SrKp7Tl','New York','50');
INSERT INTO license VALUES (DEFAULT,'8H5gCz7uaaIdk1g','Wisconsin','90');
INSERT INTO license VALUES (DEFAULT,'AmGYgulOeHNmvpo','Pennsylvania','42');
INSERT INTO license VALUES (DEFAULT,'xK295Zo3Th5Xs0n','California','30');
INSERT INTO license VALUES (DEFAULT,'S9Y6cWURiruHpue','Minnesota','51');
INSERT INTO license VALUES (DEFAULT,'DWn3EMbfNNq0vRj','Florida','75');
INSERT INTO license VALUES (DEFAULT,'Wo7SVxirXs4vnwa','Arizona','51');
INSERT INTO license VALUES (DEFAULT,'xRfLOw6bOJoawaL','South Dakota','97');
INSERT INTO license VALUES (DEFAULT,'Y0JDriFMregtjaP','Montana','55');
INSERT INTO license VALUES (DEFAULT,'dR6fI2B1Go4QkSt','Missouri','74');
INSERT INTO license VALUES (DEFAULT,'YkGSFsa8gSZrpsY','Alaska','52');
INSERT INTO license VALUES (DEFAULT,'63scP8IMl3L5GMx','Tennessee','91');
INSERT INTO license VALUES (DEFAULT,'BoeHXjggmjmJb6f','Indiana','42');
INSERT INTO license VALUES (DEFAULT,'V0YMJm7GeyODZFm','Arkansas','81');
INSERT INTO license VALUES (DEFAULT,'ruVPiUL5nYgCK7q','New York','89');
INSERT INTO license VALUES (DEFAULT,'06Or2BgA3uRdMd9','North Dakota','74');
INSERT INTO license VALUES (DEFAULT,'OuE8xgMQfObEU1i','Idaho','64');
INSERT INTO license VALUES (DEFAULT,'Pp1DzszPpj4cb4g','Ohio','61');
INSERT INTO license VALUES (DEFAULT,'t5Z3DaBaWxE6j3s','Nevada','73');
INSERT INTO license VALUES (DEFAULT,'OUMlZCPtH6SyET5','Arkansas','87');
INSERT INTO license VALUES (DEFAULT,'HnJnLPznc5vSGqL','Georgia','82');
INSERT INTO license VALUES (DEFAULT,'pMv2U3KJ5dtmRdJ','Alaska','78');
INSERT INTO license VALUES (DEFAULT,'vFvBCEX86GebBPG','Arizona','46');
INSERT INTO license VALUES (DEFAULT,'osT7xt8dPvGALcl','New Mexico','52');
INSERT INTO license VALUES (DEFAULT,'5FvunHY3Xwqs4Nm','Pennsylvania','98');
INSERT INTO license VALUES (DEFAULT,'zahv7VwhwTQ8vo2','North Carolina','73');
INSERT INTO license VALUES (DEFAULT,'UBfhAl5EZ7SvVhd','South Dakota','38');
INSERT INTO license VALUES (DEFAULT,'nL4l8t8N0yI5mIx','Kentucky','43');
INSERT INTO license VALUES (DEFAULT,'3lzbyymrhYewbvw','Maine','67');
INSERT INTO license VALUES (DEFAULT,'BrFCWRgTKxxIxye','Connecticut','32');
INSERT INTO license VALUES (DEFAULT,'o2ztyWS4aVebZDV','West Virginia','28');
INSERT INTO license VALUES (DEFAULT,'HGnMhTTInBziuk9','Texas','49');
INSERT INTO license VALUES (DEFAULT,'G4d0MUxI56C2vtb','Georgia','80');
INSERT INTO license VALUES (DEFAULT,'rYY3GjflKUcWpuf','South Carolina','30');
INSERT INTO license VALUES (DEFAULT,'6aIM4v0ElWRP2SX','Connecticut','25');
INSERT INTO license VALUES (DEFAULT,'3lVa9U7OBAUvPLh','Wyoming','76');
INSERT INTO license VALUES (DEFAULT,'xAXhwmhzfMOVSCF','Delaware','74');
INSERT INTO license VALUES (DEFAULT,'13Vn6GL82vgSVKu','Arkansas','38');
INSERT INTO license VALUES (DEFAULT,'eWncG8xleIt3t7O','Indiana','68');
INSERT INTO license VALUES (DEFAULT,'UnyD04Mj30BgJBe','Arizona','33');
INSERT INTO license VALUES (DEFAULT,'htrNv5iBb7cEJyR','Alaska','81');
INSERT INTO license VALUES (DEFAULT,'0HQHoba3MIvEVGO','Tennessee','70');
INSERT INTO license VALUES (DEFAULT,'ZftXjrQBY6ueuCu','South Dakota','53');
INSERT INTO license VALUES (DEFAULT,'KW993RQTiww7G1N','New Mexico','64');
INSERT INTO license VALUES (DEFAULT,'YwHYRZ9FPeqZcGB','Vermont','96');
INSERT INTO license VALUES (DEFAULT,'51nmgNdDA5eESFh','Oregon','40');
INSERT INTO license VALUES (DEFAULT,'n7i93793nxk4Fqu','Michigan','35');
INSERT INTO license VALUES (DEFAULT,'5Vy42jnVQefA7ZJ','New Hampshire','43');
INSERT INTO license VALUES (DEFAULT,'cE08C4ns2Z3I5Ot','Georgia','35');
INSERT INTO license VALUES (DEFAULT,'KLOSvY4nmde2pYa','Michigan','87');
INSERT INTO license VALUES (DEFAULT,'NqqK3aKL9JuMVPw','New York','85');
INSERT INTO license VALUES (DEFAULT,'P7I6rq6UZbJzNBc','Montana','30');
INSERT INTO license VALUES (DEFAULT,'BCz8RuDGwOKbQNr','Vermont','88');
INSERT INTO license VALUES (DEFAULT,'JdZ3gAu0d9BWvHR','Montana','45');
INSERT INTO license VALUES (DEFAULT,'Ouo07MF05bw2INc','Indiana','51');
INSERT INTO license VALUES (DEFAULT,'29LKeBFIxuBuAG2','Alaska','98');
INSERT INTO license VALUES (DEFAULT,'96uALQUDkj77Uwj','Hawaii','55');
INSERT INTO license VALUES (DEFAULT,'ORZGttisKbDF4V4','Connecticut','36');
INSERT INTO license VALUES (DEFAULT,'NfpVrt1G3cHKd8c','Mississippi','68');
INSERT INTO license VALUES (DEFAULT,'1kb2QNMRByR7qOD','Nevada','38');
INSERT INTO license VALUES (DEFAULT,'jEPDhii9zRbhqdk','Arizona','49');
INSERT INTO license VALUES (DEFAULT,'dTlQ7RdDq4kJNND','Virginia','47');
INSERT INTO license VALUES (DEFAULT,'YYigUB33a6vuUUq','New Jersey','51');
INSERT INTO license VALUES (DEFAULT,'uuT4mww90n70lDX','Iowa','53');
INSERT INTO license VALUES (DEFAULT,'FLsxIRrXgUsTWk3','Florida','89');
INSERT INTO license VALUES (DEFAULT,'cT6XwmwheY5Cz3X','Florida','95');
INSERT INTO license VALUES (DEFAULT,'Pe8ilclhFbDqeDE','Illinois','79');
INSERT INTO license VALUES (DEFAULT,'kEVASj5Qo3wUrbU','Alaska','55');
INSERT INTO license VALUES (DEFAULT,'UYeT00U0qKt09U0','Colorado','31');
INSERT INTO license VALUES (DEFAULT,'TTI674CS7jyMY9s','Alaska','31');
INSERT INTO license VALUES (DEFAULT,'tp3atNAJlZMZpj6','New Jersey','69');
INSERT INTO license VALUES (DEFAULT,'IBscXdcA3RfnRKA','Rhode Island','92');
INSERT INTO license VALUES (DEFAULT,'lKuojFHwVfz3R92','Wisconsin','83');
INSERT INTO license VALUES (DEFAULT,'1rImVMbaz6egUOM','Vermont','80');
INSERT INTO license VALUES (DEFAULT,'7v66i6kpC7mHmRm','Missouri','61');
INSERT INTO license VALUES (DEFAULT,'TS1RP8SOuInQBRn','New York','64');
INSERT INTO license VALUES (DEFAULT,'BHj0Svi7TOzTNp1','Vermont','86');
INSERT INTO license VALUES (DEFAULT,'43hdswnsyx7ZBq8','Tennessee','31');
INSERT INTO license VALUES (DEFAULT,'PehoAhTU6ud5hdv','Alaska','39');
INSERT INTO license VALUES (DEFAULT,'Wu6ETV6He7kiRh1','South Dakota','34');
INSERT INTO license VALUES (DEFAULT,'ZW2K3VNIvCKxRTL','Michigan','86');
INSERT INTO license VALUES (DEFAULT,'TOqA5JkggjswoMl','Virginia','54');
INSERT INTO license VALUES (DEFAULT,'EY8BdYs5iSwUNoZ','Rhode Island','59');
INSERT INTO license VALUES (DEFAULT,'Om9HdNRnXweWGzi','Michigan','90');
INSERT INTO license VALUES (DEFAULT,'Pxjx008FX0yMT5T','Maryland','39');
INSERT INTO license VALUES (DEFAULT,'89QRkLrXGlNibdN','Maryland','49');
INSERT INTO license VALUES (DEFAULT,'ZlboYwABGWcaD1L','Montana','92');
INSERT INTO license VALUES (DEFAULT,'1jSlom1ZtOug5Xl','Indiana','97');
INSERT INTO license VALUES (DEFAULT,'frUhjMPPeyP7CuC','Ohio','64');
INSERT INTO license VALUES (DEFAULT,'lodtu9jOohQfK81','Rhode Island','30');
INSERT INTO license VALUES (DEFAULT,'VdPWRWDVKxmcrer','Michigan','54');
INSERT INTO license VALUES (DEFAULT,'w66PvsvS5p8uT0Y','Massachusetts','97');
INSERT INTO license VALUES (DEFAULT,'YjsKTSaYjL7v7Dj','Massachusetts','81');
INSERT INTO license VALUES (DEFAULT,'M4DyjueDWqEk06O','Wisconsin','78');
INSERT INTO license VALUES (DEFAULT,'FH0Ocog1Mgxh8mY','Mississippi','65');
INSERT INTO license VALUES (DEFAULT,'od3JzzNG7lVxsOt','Virginia','42');
INSERT INTO license VALUES (DEFAULT,'7bGhuQh7XNqgYEp','Louisiana','47');
INSERT INTO license VALUES (DEFAULT,'SumjQSmHeDKDdRE','Iowa','79');
INSERT INTO license VALUES (DEFAULT,'o7rA4NWLclgkjgq','Rhode Island','55');
INSERT INTO license VALUES (DEFAULT,'69nprJdaPMX335m','Virginia','73');
INSERT INTO license VALUES (DEFAULT,'asFSYgQXxqGh5ge','Oregon','30');
INSERT INTO license VALUES (DEFAULT,'VFoD7xNkoA8Rjp4','Delaware','92');
INSERT INTO license VALUES (DEFAULT,'G75HhcWS3cX2Tq3','Alaska','85');
INSERT INTO license VALUES (DEFAULT,'jgkld51rufRFOAL','Kansas','26');
INSERT INTO license VALUES (DEFAULT,'ahI6ymRvDU0rUUF','Indiana','80');
INSERT INTO license VALUES (DEFAULT,'FXuTmBDZx7EMpcL','New Jersey','47');
INSERT INTO license VALUES (DEFAULT,'PWs1v3TzZlAAZKM','Colorado','60');
INSERT INTO license VALUES (DEFAULT,'4xcrxfEhBpNiY2o','Rhode Island','27');
INSERT INTO license VALUES (DEFAULT,'Ets2UeadMaCZYw2','Michigan','38');
INSERT INTO license VALUES (DEFAULT,'hXOlb0oUtNGRBsW','North Carolina','52');
INSERT INTO license VALUES (DEFAULT,'Gc5lkTBOlEBHn3U','North Carolina','40');
INSERT INTO license VALUES (DEFAULT,'UQwan2EhkQUMLrq','Vermont','26');
INSERT INTO license VALUES (DEFAULT,'qZnj7EMW0vmMGTv','Arizona','85');
INSERT INTO license VALUES (DEFAULT,'XFN6ck1D9CqBZWU','South Carolina','70');
INSERT INTO license VALUES (DEFAULT,'N4WnXMjEwGbVe6j','Hawaii','52');
INSERT INTO license VALUES (DEFAULT,'TlMENzwf7BgUaZS','New York','60');
INSERT INTO license VALUES (DEFAULT,'G04KA3n1qdZseFH','Missouri','28');
INSERT INTO license VALUES (DEFAULT,'DtfhAFkHZKDtVrJ','Alaska','25');
INSERT INTO license VALUES (DEFAULT,'0sAMTsiNgWWxd6Y','Washington','53');
INSERT INTO license VALUES (DEFAULT,'Vi86FvTCkskIVvQ','Alabama','61');
INSERT INTO license VALUES (DEFAULT,'PZG0tA6UetOU6Op','Wisconsin','96');
INSERT INTO license VALUES (DEFAULT,'Vk1Dv7qJt98ELkG','Hawaii','97');
INSERT INTO license VALUES (DEFAULT,'q8HhmiBmlCFrne5','Tennessee','52');
INSERT INTO license VALUES (DEFAULT,'Zp9aecZq9B8PSII','Vermont','96');
INSERT INTO license VALUES (DEFAULT,'hqXasC0aJ4kbGfD','New Mexico','84');
INSERT INTO license VALUES (DEFAULT,'Ka5hMgr0ngDXS37','Maine','74');
INSERT INTO license VALUES (DEFAULT,'6OP6rnJ9UhzO2g5','Mississippi','61');
INSERT INTO license VALUES (DEFAULT,'7rdaVKqkFfxD1An','California','57');
INSERT INTO license VALUES (DEFAULT,'iyL1dxwh182luAI','Idaho','84');
INSERT INTO license VALUES (DEFAULT,'6QzOlEjWlHW6dgM','Arkansas','58');
INSERT INTO license VALUES (DEFAULT,'7Gmnd4r2dLx8Ueo','Wisconsin','28');
INSERT INTO license VALUES (DEFAULT,'EWTM6tme1pZlYbq','Wisconsin','47');
INSERT INTO license VALUES (DEFAULT,'DB6KcVIWI3vXRmr','Indiana','56');
INSERT INTO license VALUES (DEFAULT,'czwPBW9IlORQEtH','Florida','27');
INSERT INTO license VALUES (DEFAULT,'bni3lALzx00GkCr','Wyoming','97');
INSERT INTO license VALUES (DEFAULT,'91tDBo6B0H0FDKo','Nevada','43');
INSERT INTO license VALUES (DEFAULT,'LE2g0ykZNgJCBLj','Arkansas','32');
INSERT INTO license VALUES (DEFAULT,'tPu8a8m1u3VutqM','Kansas','50');
INSERT INTO license VALUES (DEFAULT,'1nSbIDInxIx9z5Q','North Carolina','95');
INSERT INTO license VALUES (DEFAULT,'aZKQvFPpF4Ppd07','Nebraska','74');
INSERT INTO license VALUES (DEFAULT,'U6wolMlVm0y258E','Texas','82');
INSERT INTO license VALUES (DEFAULT,'PwBFGiCwCI27US0','California','37');
INSERT INTO license VALUES (DEFAULT,'kUMSzMi4JPbwK6K','Hawaii','76');
INSERT INTO license VALUES (DEFAULT,'kKyjOYt2iDsXRh8','Kentucky','80');
INSERT INTO license VALUES (DEFAULT,'XuhdakcK6SlA1Ms','Oregon','71');
INSERT INTO license VALUES (DEFAULT,'rqVkW6caQJ0h1rc','Connecticut','55');
INSERT INTO license VALUES (DEFAULT,'jaXNKjVYRFI2q1A','Delaware','73');

INSERT INTO discount VALUES (DEFAULT,'0','Default ');
INSERT INTO discount VALUES (DEFAULT,'73','Description for discount with ID 1');
INSERT INTO discount VALUES (DEFAULT,'50','Description for discount with ID 2');
INSERT INTO discount VALUES (DEFAULT,'34','Description for discount with ID 3');
INSERT INTO discount VALUES (DEFAULT,'57','Description for discount with ID 4');
INSERT INTO discount VALUES (DEFAULT,'33','Description for discount with ID 5');
INSERT INTO discount VALUES (DEFAULT,'72','Description for discount with ID 6');
INSERT INTO discount VALUES (DEFAULT,'53','Description for discount with ID 7');
INSERT INTO discount VALUES (DEFAULT,'34','Description for discount with ID 8');
INSERT INTO discount VALUES (DEFAULT,'60','Description for discount with ID 9');
INSERT INTO discount VALUES (DEFAULT,'21','Description for discount with ID 10');
INSERT INTO discount VALUES (DEFAULT,'12','Description for discount with ID 11');
INSERT INTO discount VALUES (DEFAULT,'73','Description for discount with ID 12');
INSERT INTO discount VALUES (DEFAULT,'9','Description for discount with ID 13');
INSERT INTO discount VALUES (DEFAULT,'14','Description for discount with ID 14');
INSERT INTO discount VALUES (DEFAULT,'17','Description for discount with ID 15');
INSERT INTO discount VALUES (DEFAULT,'18','Description for discount with ID 16');
INSERT INTO discount VALUES (DEFAULT,'47','Description for discount with ID 17');
INSERT INTO discount VALUES (DEFAULT,'9','Description for discount with ID 18');
INSERT INTO discount VALUES (DEFAULT,'49','Description for discount with ID 19');

INSERT INTO membership VALUES (DEFAULT,'Default','1');
INSERT INTO membership VALUES (DEFAULT,'CarMax','14');
INSERT INTO membership VALUES (DEFAULT,'Amazon.com','13');
INSERT INTO membership VALUES (DEFAULT,'Jackbox','4');
INSERT INTO membership VALUES (DEFAULT,'DynCorp','8');
INSERT INTO membership VALUES (DEFAULT,'Zepter','16');
INSERT INTO membership VALUES (DEFAULT,'Bars','1');
INSERT INTO membership VALUES (DEFAULT,'Erickson','16');
INSERT INTO membership VALUES (DEFAULT,'Demaco','12');
INSERT INTO membership VALUES (DEFAULT,'Facebook','6');
INSERT INTO membership VALUES (DEFAULT,'UPC','7');
INSERT INTO membership VALUES (DEFAULT,'Podaphone','3');
INSERT INTO membership VALUES (DEFAULT,'Areon Impex','7');
INSERT INTO membership VALUES (DEFAULT,'LTT','2');
INSERT INTO membership VALUES (DEFAULT,'Apple Inc.','15');
INSERT INTO membership VALUES (DEFAULT,'It Smart Group','4');
INSERT INTO membership VALUES (DEFAULT,'Coca-Cola Company','19');
INSERT INTO membership VALUES (DEFAULT,'Duluth','7');
INSERT INTO membership VALUES (DEFAULT,'Vodafone','16');
INSERT INTO membership VALUES (DEFAULT,'Yarp','15');
INSERT INTO membership VALUES (DEFAULT,'Mars','8');
INSERT INTO membership VALUES (DEFAULT,'Team Guard SRL','8');
INSERT INTO membership VALUES (DEFAULT,'Biolife Grup','13');
INSERT INTO membership VALUES (DEFAULT,'Blarkfel','2');
INSERT INTO membership VALUES (DEFAULT,'Jarmen','2');

INSERT INTO customer VALUES (DEFAULT,'Mr. Cedrick Carter, Lonsdale 3222, Prague - 4432, Belarus','Cedrick Carter','50','15');
INSERT INTO customer VALUES (DEFAULT,'Mr. Julius Uddin, Clavell 4789, Chicago - 6587, Ireland','Julius Uddin','76','8');
INSERT INTO customer VALUES (DEFAULT,'Mr. Maxwell Mullins, King Edward 5517, Kansas City - 1360, Bhutan','Maxwell Mullins','63','22');
INSERT INTO customer VALUES (DEFAULT,'Miss Carmella Vince, Antrobus 4998, Baltimore - 8634, Malawi','Carmella Vince','97','15');
INSERT INTO customer VALUES (DEFAULT,'Mr. Mason Jones, North 5746, Amarillo - 1125, Vatican City','Mason Jones','92','23');
INSERT INTO customer VALUES (DEFAULT,'Mr. Harvey Crawford, Chamberlain 563, Seattle - 2555, The Bahamas','Harvey Crawford','22','4');
INSERT INTO customer VALUES (DEFAULT,'Ms. Marla Burnley, Oxford 8289, Lincoln - 1513, Gabon','Marla Burnley','24','3');
INSERT INTO customer VALUES (DEFAULT,'Miss Diane Johnson, Bacton 8406, Jersey City - 1086, Honduras','Diane Johnson','66','17');
INSERT INTO customer VALUES (DEFAULT,'Miss Gladys Salt, Howard 2644, Indianapolis - 8074, United Kingdom','Gladys Salt','72','6');
INSERT INTO customer VALUES (DEFAULT,'Mr. Manuel Wren, Belmore 4883, Otawa - 8347, South Africa','Manuel Wren','6','20');
INSERT INTO customer VALUES (DEFAULT,'Mr. Michael Zaoui, Sundown 4418, Tulsa - 1836, Yemen','Michael Zaoui','16','11');
INSERT INTO customer VALUES (DEFAULT,'Mr. Nate Seymour, Cliffords 5240, Fremont - 3088, Malta','Nate Seymour','13','7');
INSERT INTO customer VALUES (DEFAULT,'Mr. Doug Foxley, Linden 7497, Honolulu - 7017, Tajikistan','Doug Foxley','67','2');
INSERT INTO customer VALUES (DEFAULT,'Mr. Owen Hancock, Cam 3647, Zurich - 4062, Thailand','Owen Hancock','47','4');
INSERT INTO customer VALUES (DEFAULT,'Miss Ciara Little, Blandford 4252, Tulsa - 8540, Ecuador','Ciara Little','31','18');
INSERT INTO customer VALUES (DEFAULT,'Mr. William Lambert, Lexington 9503, Zurich - 7455, Seychelles','William Lambert','17','18');
INSERT INTO customer VALUES (DEFAULT,'Mr. Bart Huggins, Viscount 4313, Lakewood - 0601, Kuwait','Bart Huggins','88','17');
INSERT INTO customer VALUES (DEFAULT,'Mr. William Wilson, Victoria 9132, El Paso - 2520, San Marino','William Wilson','69','14');
INSERT INTO customer VALUES (DEFAULT,'Mr. Tony Mackenzie, Glenwood 4505, Springfield - 6618, Uganda','Tony Mackenzie','62','6');
INSERT INTO customer VALUES (DEFAULT,'Ms. Kimberly Ranks, Norfolk 9278, Dallas - 4776, China','Kimberly Ranks','30','20');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Stephanie Rainford, Balfour 2643, Springfield - 5871, Myanmar','Stephanie Rainford','20','6');
INSERT INTO customer VALUES (DEFAULT,'Ms. Sofia Eddison, Castle 4491, Jersey City - 2002, Libya','Sofia Eddison','48','13');
INSERT INTO customer VALUES (DEFAULT,'Mr. Johnathan Rose, Geary 5527, San Francisco - 5341, Tanzania','Johnathan Rose','60','10');
INSERT INTO customer VALUES (DEFAULT,'Mr. Manuel Downing, Cave 2676, Philadelphia - 3686, United Arab Emirates','Manuel Downing','23','23');
INSERT INTO customer VALUES (DEFAULT,'Mr. Nathan Tait, Caldwell 3703, London - 8430, Malawi','Nathan Tait','42','4');
INSERT INTO customer VALUES (DEFAULT,'Mr. Fred Bell, Cavendish 5035, Hollywood - 8518, Uzbekistan','Fred Bell','53','19');
INSERT INTO customer VALUES (DEFAULT,'Miss Samara Price, Central 8200, Salt Lake City - 5874, Italy','Samara Price','59','20');
INSERT INTO customer VALUES (DEFAULT,'Mr. Tom Richardson, Chicksand 2169, Wien - 8788, Fiji','Tom Richardson','51','13');
INSERT INTO customer VALUES (DEFAULT,'Ms. Hope Kelly, Gavel 7636, Tallahassee - 0318, Kenya','Hope Kelly','41','10');
INSERT INTO customer VALUES (DEFAULT,'Mr. Michael Trent, Baylis 4856, Bakersfield - 6177, Bangladesh','Michael Trent','81','12');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Celia Hobson, Antrobus 9819, Charlotte - 5745, Samoa','Celia Hobson','58','1');
INSERT INTO customer VALUES (DEFAULT,'Ms. Celia Morgan, Magnolia 4533, Santa Ana - 3022, Israel','Celia Morgan','98','9');
INSERT INTO customer VALUES (DEFAULT,'Mr. Chadwick Thorne, Angrave 1638, Glendale - 3536, Seychelles','Chadwick Thorne','27','23');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Vicky Asher, Linden 3230, Venice - 2877, Jamaica','Vicky Asher','44','19');
INSERT INTO customer VALUES (DEFAULT,'Mr. Cedrick Patel, Adams 4193, San Diego - 1527, Burkina Faso','Cedrick Patel','46','12');
INSERT INTO customer VALUES (DEFAULT,'Mr. Oliver Moore, Cliffords 2218, Richmond - 4142, Philippines','Oliver Moore','26','5');
INSERT INTO customer VALUES (DEFAULT,'Mr. Bryon Coleman, Cleaver 380, Bridgeport - 5533, Mali','Bryon Coleman','79','7');
INSERT INTO customer VALUES (DEFAULT,'Mr. Roger Fields, Bendmore 5215, Indianapolis - 5021, Sudan, South','Roger Fields','32','8');
INSERT INTO customer VALUES (DEFAULT,'Miss Sonya Walsh, Collent 2752, Denver - 3542, Equatorial Guinea','Sonya Walsh','29','2');
INSERT INTO customer VALUES (DEFAULT,'Mr. Hayden Sawyer, Under 1823, Richmond - 7186, Belgium','Hayden Sawyer','68','11');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Brooklyn Heaton, Pine 6547, Columbus - 2131, Cuba','Brooklyn Heaton','71','4');
INSERT INTO customer VALUES (DEFAULT,'Ms. Greta Brooks, Apostle 116, Escondido - 3768, Monaco','Greta Brooks','91','10');
INSERT INTO customer VALUES (DEFAULT,'Mr. Alexander Grant, Ernest 3788, Atlanta - 5487, Libya','Alexander Grant','87','7');
INSERT INTO customer VALUES (DEFAULT,'Miss Camila Jackson, Blackpool 474, Omaha - 8670, Bolivia','Camila Jackson','83','18');
INSERT INTO customer VALUES (DEFAULT,'Ms. Kimberly Pearson, Fawn 3363, Hayward - 1060, Guyana','Kimberly Pearson','55','4');
INSERT INTO customer VALUES (DEFAULT,'Mr. Boris Rogan, Fairholt 5842, Omaha - 5016, Czech Republic','Boris Rogan','2','8');
INSERT INTO customer VALUES (DEFAULT,'Mr. Bryon Edmonds, Railroad 9624, Arlington - 6625, Zimbabwe','Bryon Edmonds','35','18');
INSERT INTO customer VALUES (DEFAULT,'Ms. Janice Cobb, Clarges 6603, Houston - 7302, Australia','Janice Cobb','74','12');
INSERT INTO customer VALUES (DEFAULT,'Ms. Elly Hastings, Bletchley 4663, Worcester - 1174, Korea, South','Elly Hastings','5','25');
INSERT INTO customer VALUES (DEFAULT,'Ms. Makenzie Gregory, Abbots 3730, Milano - 2178, South Africa','Makenzie Gregory','12','15');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Faith West, Bury 9550, Minneapolis - 5027, Argentina','Faith West','18','20');
INSERT INTO customer VALUES (DEFAULT,'Ms. Doris Gray, Battis 4850, Stockton - 0762, Norway','Doris Gray','100','13');
INSERT INTO customer VALUES (DEFAULT,'Ms. Cadence Adams, Ampton 9818, Toledo - 0834, Swaziland','Cadence Adams','10','19');
INSERT INTO customer VALUES (DEFAULT,'Ms. Rachael Bell, Colombo 8871, Memphis - 5271, Germany','Rachael Bell','45','5');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Bristol Purvis, Cave 8932, Colorado Springs - 8147, Gabon','Bristol Purvis','49','21');
INSERT INTO customer VALUES (DEFAULT,'Mr. Jayden Moore, Wakley 7682, Madrid - 4643, Panama','Jayden Moore','19','7');
INSERT INTO customer VALUES (DEFAULT,'Miss Greta Russell, Camelot 6561, Santa Ana - 6480, Liechtenstein','Greta Russell','7','19');
INSERT INTO customer VALUES (DEFAULT,'Ms. Alice Flett, Cardinal 5526, Moreno Valley - 3204, Botswana','Alice Flett','78','19');
INSERT INTO customer VALUES (DEFAULT,'Mr. Elijah Allcott, Arbutus 6383, Bridgeport - 4788, Nepal','Elijah Allcott','75','6');
INSERT INTO customer VALUES (DEFAULT,'Mr. Denis Young, Cave 918, Henderson - 6255, Poland','Denis Young','40','9');
INSERT INTO customer VALUES (DEFAULT,'Ms. Lara Dempsey, Champion 9848, Zurich - 5430, Italy','Lara Dempsey','64','25');
INSERT INTO customer VALUES (DEFAULT,'Miss Dalia Jarvis, Babmaes 5321, New Orleans - 4446, Mozambique','Dalia Jarvis','77','23');
INSERT INTO customer VALUES (DEFAULT,'Ms. Jessica Sherry, Lake 7396, Huntsville - 2560, Sri Lanka','Jessica Sherry','36','20');
INSERT INTO customer VALUES (DEFAULT,'Miss Cassidy Graham, Elba 2933, Minneapolis - 7722, Myanmar','Cassidy Graham','94','19');
INSERT INTO customer VALUES (DEFAULT,'Miss Dorothy Partridge, Egerton 1799, Laredo - 8600, Taiwan','Dorothy Partridge','89','7');
INSERT INTO customer VALUES (DEFAULT,'Miss Natalie Davies, Abbots 8266, Irving - 4221, Brazil','Natalie Davies','1','12');
INSERT INTO customer VALUES (DEFAULT,'Ms. Cynthia Wild, Blean 8343, Lisbon - 7813, Belarus','Cynthia Wild','65','8');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Peyton Allen, Hammersmith 659, Minneapolis - 6334, Myanmar','Peyton Allen','34','24');
INSERT INTO customer VALUES (DEFAULT,'Mr. Rick Murray, Bacon 8482, Toledo - 3068, Trinidad and Tobago','Rick Murray','39','16');
INSERT INTO customer VALUES (DEFAULT,'Miss Jade Gunn, Linda Lane 3043, Atlanta - 1684, Argentina','Jade Gunn','86','9');
INSERT INTO customer VALUES (DEFAULT,'Ms. Josephine Gibbons, King 3507, Minneapolis - 3135, Senegal','Josephine Gibbons','8','12');
INSERT INTO customer VALUES (DEFAULT,'Ms. Gina Miller, Aylward 386, Irving - 5260, Lebanon','Gina Miller','15','6');
INSERT INTO customer VALUES (DEFAULT,'Mr. Chris Oakley, Woodland 3647, Pittsburgh - 6102, Jamaica','Chris Oakley','21','23');
INSERT INTO customer VALUES (DEFAULT,'Mr. Alexander Drew, Cheltenham 5733, Innsbruck - 7213, Monaco','Alexander Drew','54','21');
INSERT INTO customer VALUES (DEFAULT,'Miss Mina Utterson, Clerkenwell 8642, Lyon - 8218, Comoros','Mina Utterson','96','2');
INSERT INTO customer VALUES (DEFAULT,'Miss Bristol Cameron, Balfour 8522, Madrid - 3771, Korea, South','Bristol Cameron','95','25');
INSERT INTO customer VALUES (DEFAULT,'Mr. Carter Everett, Lake 1883, Long Beach - 0178, Taiwan','Carter Everett','99','12');
INSERT INTO customer VALUES (DEFAULT,'Mr. Ramon Williams, Birkin 2961, Bakersfield - 7781, Central African Republic','Ramon Williams','73','6');
INSERT INTO customer VALUES (DEFAULT,'Mr. Tyson Dwyer, Bush 9732, Richmond - 2260, Romania','Tyson Dwyer','38','8');
INSERT INTO customer VALUES (DEFAULT,'Mr. Johnny Gordon, Bennett 8548, Madrid - 7104, Czech Republic','Johnny Gordon','57','22');
INSERT INTO customer VALUES (DEFAULT,'Mr. Henry Yoman, Ely 3487, Valetta - 6606, Cuba','Henry Yoman','28','13');
INSERT INTO customer VALUES (DEFAULT,'Ms. Violet Emmott, Virginia 7166, Albuquerque - 5825, Japan','Violet Emmott','90','6');
INSERT INTO customer VALUES (DEFAULT,'Ms. Gina Tutton, Birkenhead 3673, Amarillo - 8831, Finland','Gina Tutton','93','6');
INSERT INTO customer VALUES (DEFAULT,'Miss Evelynn Walsh, Bletchley 3592, Rome - 5548, USA','Evelynn Walsh','37','19');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Daria Ellery, Blandford 8535, Philadelphia - 0356, Sudan, South','Daria Ellery','70','20');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Tania Harrison, Queensberry 7372, Toledo - 6644, Nicaragua','Tania Harrison','14','10');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Mara Nanton, Glenwood 982, Toledo - 3084, Honduras','Mara Nanton','61','1');
INSERT INTO customer VALUES (DEFAULT,'Mr. Chad Wren, Beechen 482, Washington - 0874, Indonesia','Chad Wren','33','15');
INSERT INTO customer VALUES (DEFAULT,'Mr. Percy Hobbs, Coleman 2311, Escondido - 0612, Senegal','Percy Hobbs','9','7');
INSERT INTO customer VALUES (DEFAULT,'Mr. Nicholas Wilton, Rosewood 3438, Scottsdale - 4881, Sudan, South','Nicholas Wilton','11','4');
INSERT INTO customer VALUES (DEFAULT,'Miss Ellen Edler, Clerkenwell 4628, Milwaukee - 3733, Kazakhstan','Ellen Edler','25','12');
INSERT INTO customer VALUES (DEFAULT,'Mr. Ethan Windsor, Carolina 4285, Murfreesboro - 1144, Antigua and Barbuda','Ethan Windsor','52','12');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Janelle Buckley, Thomas More 9367, Bucharest - 2862, Liberia','Janelle Buckley','3','2');
INSERT INTO customer VALUES (DEFAULT,'Ms. Tiffany Pond, Eaglet 9490, Oakland - 1875, Pakistan','Tiffany Pond','4','25');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Anne Cartwright, Baynes 502, San Diego - 1300, Australia','Anne Cartwright','43','6');
INSERT INTO customer VALUES (DEFAULT,'Mr. Luke Wright, Apothecary 1422, Fremont - 8858, Bahrain','Luke Wright','80','10');
INSERT INTO customer VALUES (DEFAULT,'Mr. Alan Stevens, Apollo 3438, Jersey City - 7312, Rwanda','Alan Stevens','82','3');
INSERT INTO customer VALUES (DEFAULT,'Miss Jamie Dempsey, Westbourne 565, Fullerton - 8553, Gabon','Jamie Dempsey','84','3');
INSERT INTO customer VALUES (DEFAULT,'Mr. Mike Walton, Dunstable 7240, San Jose - 6787, Antigua and Barbuda','Mike Walton','56','22');
INSERT INTO customer VALUES (DEFAULT,'Mr. Chadwick Ranks, Woodland 1789, Detroit - 6157, United Arab Emirates','Chadwick Ranks','175','9');
INSERT INTO customer VALUES (DEFAULT,'Miss Sara Kennedy, Blendon 9974, Valetta - 0733, USA','Sara Kennedy','110','18');
INSERT INTO customer VALUES (DEFAULT,'Mr. Nick Edley, Bendmore 5150, Oakland - 1567, Côte d’Ivoire','Nick Edley','149','23');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Michaela Richardson, Calverley 6289, Murfreesboro - 2884, Vanuatu','Michaela Richardson','53','6');
INSERT INTO customer VALUES (DEFAULT,'Mr. Clint Campbell, Gautrey 8440, Escondido - 6035, Liechtenstein','Clint Campbell','139','11');
INSERT INTO customer VALUES (DEFAULT,'Mr. Barry Preston, Apsley 7551, Tallahassee - 5781, Peru','Barry Preston','141','16');
INSERT INTO customer VALUES (DEFAULT,'Mr. Ramon Lewis, Howard 801, Salt Lake City - 4815, Romania','Ramon Lewis','28','18');
INSERT INTO customer VALUES (DEFAULT,'Miss Kirsten Olivier, Western 2074, Baltimore - 2268, Iran','Kirsten Olivier','131','5');
INSERT INTO customer VALUES (DEFAULT,'Mr. Liam Antcliff, Castlereagh 8636, Anaheim - 0227, Eritrea','Liam Antcliff','89','5');
INSERT INTO customer VALUES (DEFAULT,'Mr. William Whatson, Badric 353, Valetta - 3552, Kuwait','William Whatson','164','6');
INSERT INTO customer VALUES (DEFAULT,'Mr. Clint Ventura, Chestnut 9133, Milano - 2064, East Timor (Timor-Leste)','Clint Ventura','169','18');
INSERT INTO customer VALUES (DEFAULT,'Mr. Anthony Carson, Parkgate 5872, Bucharest - 0811, Bhutan','Anthony Carson','82','23');
INSERT INTO customer VALUES (DEFAULT,'Ms. Violet Wade, Cam 5515, Las Vegas - 0530, Cabo Verde','Violet Wade','21','12');
INSERT INTO customer VALUES (DEFAULT,'Mr. Ramon Miller, Woodland 6121, Springfield - 7472, Jamaica','Ramon Miller','147','23');
INSERT INTO customer VALUES (DEFAULT,'Miss Stacy Hall, Cockspur 1349, Hollywood - 0272, Nigeria','Stacy Hall','67','12');
INSERT INTO customer VALUES (DEFAULT,'Mr. Alan Walsh, Apollo 4553, Nashville - 2864, Israel','Alan Walsh','12','4');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Blake Oldfield, Longman 4303, Lakewood - 4560, Somalia','Blake Oldfield','84','16');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Julia Dillon, Blackheath 220, Philadelphia - 7664, Afghanistan','Julia Dillon','184','15');
INSERT INTO customer VALUES (DEFAULT,'Ms. Hannah Powell, Charterhouse 9891, Madrid - 4712, Vietnam','Hannah Powell','51','22');
INSERT INTO customer VALUES (DEFAULT,'Miss Cameron Thorpe, Gavel 7628, Bridgeport - 1106, Niger','Cameron Thorpe','126','21');
INSERT INTO customer VALUES (DEFAULT,'Miss Hailey Lambert, Chancellor 6919, Detroit - 8056, Ireland','Hailey Lambert','181','4');
INSERT INTO customer VALUES (DEFAULT,'Mr. Denis Walker, Bletchley 5935, Bellevue - 5635, Kenya','Denis Walker','5','16');
INSERT INTO customer VALUES (DEFAULT,'Miss Rosalee Rigg, Endsleigh 3161, Berna - 5110, Brazil','Rosalee Rigg','31','4');
INSERT INTO customer VALUES (DEFAULT,'Mr. Peter Appleton, Belgrave 3883, Charlotte - 2743, Guatemala','Peter Appleton','17','14');
INSERT INTO customer VALUES (DEFAULT,'Mr. William Gibbons, Tisbury 1982, Henderson - 5101, Kenya','William Gibbons','115','12');
INSERT INTO customer VALUES (DEFAULT,'Miss Hannah Warren, Birkbeck 1867, Otawa - 2714, Togo','Hannah Warren','4','25');
INSERT INTO customer VALUES (DEFAULT,'Mr. William Moss, Cleveland 4132, Bucharest - 1551, El Salvador','William Moss','177','11');
INSERT INTO customer VALUES (DEFAULT,'Miss Hannah Garcia, Bolingbroke 4704, Lancaster - 1272, Dominican Republic','Hannah Garcia','85','17');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Lucy Bennett, Chandos 1800, Minneapolis - 5505, Kenya','Lucy Bennett','166','23');
INSERT INTO customer VALUES (DEFAULT,'Mr. Denis Verdon, Dutton 9776, Memphis - 2673, Lesotho','Denis Verdon','135','5');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Mavis Crawford, Sheffield 3496, Omaha - 0767, Liberia','Mavis Crawford','18','1');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Karla Goodman, Lincoln 4724, Miami - 3288, Italy','Karla Goodman','44','8');
INSERT INTO customer VALUES (DEFAULT,'Ms. Michaela Flack, Hickory 9919, Cincinnati - 5761, Fiji','Michaela Flack','144','2');
INSERT INTO customer VALUES (DEFAULT,'Ms. Julia Farrow, Archery 5419, Lisbon - 2063, Hungary','Julia Farrow','197','4');
INSERT INTO customer VALUES (DEFAULT,'Mr. Liam Santos, Blackheath 8992, Fayetteville - 5788, Bolivia','Liam Santos','23','10');
INSERT INTO customer VALUES (DEFAULT,'Miss Zara Moreno, Cloth 1770, Saint Paul - 6133, Angola','Zara Moreno','9','14');
INSERT INTO customer VALUES (DEFAULT,'Mr. Sebastian Clark, Chantry 6858, Laredo - 6436, Poland','Sebastian Clark','193','24');
INSERT INTO customer VALUES (DEFAULT,'Miss Hope Nicholls, Fawn 6243, Kansas City - 0038, Azerbaijan','Hope Nicholls','118','4');
INSERT INTO customer VALUES (DEFAULT,'Miss Bristol Tennant, Carltoun 6407, Atlanta - 1458, Andorra','Bristol Tennant','196','13');
INSERT INTO customer VALUES (DEFAULT,'Mr. Denny Burnley, Bicknell 1661, Richmond - 7476, Bangladesh','Denny Burnley','154','13');
INSERT INTO customer VALUES (DEFAULT,'Mr. Johnathan Price, Waite 3429, Salem - 8266, Hungary','Johnathan Price','100','8');
INSERT INTO customer VALUES (DEFAULT,'Miss Stacy Bristow, Ellerman 5490, Salem - 4408, Honduras','Stacy Bristow','161','23');
INSERT INTO customer VALUES (DEFAULT,'Mr. Bart Torres, Burton 1147, Pittsburgh - 2458, Liechtenstein','Bart Torres','99','20');
INSERT INTO customer VALUES (DEFAULT,'Mr. Anthony Cooper, Charnwood 9890, El Paso - 1606, Tajikistan','Anthony Cooper','148','11');
INSERT INTO customer VALUES (DEFAULT,'Miss Alessandra Dixon, Clifton 2774, Tokyo - 8675, Maldives','Alessandra Dixon','58','8');
INSERT INTO customer VALUES (DEFAULT,'Mr. Daron Wellington, Sheringham 4462, Boston - 7322, Syria','Daron Wellington','160','19');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Aurelia Glynn, Queensberry 8316, Saint Paul - 8363, USA','Aurelia Glynn','90','17');
INSERT INTO customer VALUES (DEFAULT,'Ms. Winnie Greenwood, Carolina 9169, Sacramento - 6370, Seychelles','Winnie Greenwood','127','7');
INSERT INTO customer VALUES (DEFAULT,'Mr. Johnathan Bell, Angrave 9037, Bakersfield - 1883, Solomon Islands','Johnathan Bell','20','11');
INSERT INTO customer VALUES (DEFAULT,'Miss Martha Walton, Fieldstone 6491, Santa Ana - 2758, Liechtenstein','Martha Walton','159','6');
INSERT INTO customer VALUES (DEFAULT,'Mr. Gabriel Miller, Cockspur 5784, Columbus - 5001, Haiti','Gabriel Miller','55','8');
INSERT INTO customer VALUES (DEFAULT,'Mr. Denny Truscott, Shepherds 2231, Wien - 2218, Micronesia','Denny Truscott','117','15');
INSERT INTO customer VALUES (DEFAULT,'Miss Amy Kent, Battersea 7305, Paris - 6663, Czech Republic','Amy Kent','86','7');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Tiffany Robinson, Chapel 7623, Orlando - 1230, Ukraine','Tiffany Robinson','42','11');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Callie Wilde, Clifton 7667, Toledo - 5003, Grenada','Callie Wilde','106','22');
INSERT INTO customer VALUES (DEFAULT,'Ms. Jackeline Morgan, Edwin 5008, Cincinnati - 1836, Cuba','Jackeline Morgan','102','7');
INSERT INTO customer VALUES (DEFAULT,'Mr. Johnathan Yarwood, Paris 8264, Hayward - 6551, Japan','Johnathan Yarwood','129','19');
INSERT INTO customer VALUES (DEFAULT,'Mr. Caleb Rodwell, Dyott 8199, Moreno Valley - 6162, Papua New Guinea','Caleb Rodwell','125','10');
INSERT INTO customer VALUES (DEFAULT,'Mr. Javier White, Unwin 9939, Fullerton - 0550, United Arab Emirates','Javier White','63','16');
INSERT INTO customer VALUES (DEFAULT,'Ms. Carrie Ward, Baylis 7282, Rochester - 3672, United Kingdom','Carrie Ward','49','19');
INSERT INTO customer VALUES (DEFAULT,'Ms. Samara Summers, Cadogan 349, Glendale - 8333, Rwanda','Samara Summers','54','22');
INSERT INTO customer VALUES (DEFAULT,'Mr. Gabriel Robe, Marina 1112, Stockton - 4420, Norway','Gabriel Robe','128','1');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Scarlett Parsons, Bayberry 9049, San Antonio - 3803, Austria','Scarlett Parsons','62','15');
INSERT INTO customer VALUES (DEFAULT,'Mr. Hayden Stuart, Cheney 797, Milano - 4236, Australia','Hayden Stuart','146','19');
INSERT INTO customer VALUES (DEFAULT,'Mr. Caleb Haines, Champion 7769, Santa Ana - 1641, Saudi Arabia','Caleb Haines','43','20');
INSERT INTO customer VALUES (DEFAULT,'Mr. Denny York, Blomfield 9948, Saint Paul - 4501, Estonia','Denny York','26','21');
INSERT INTO customer VALUES (DEFAULT,'Mr. Clint Spencer, Cave 4973, Santa Ana - 4726, USA','Clint Spencer','104','16');
INSERT INTO customer VALUES (DEFAULT,'Ms. Maia Parker, Thomas More 2850, Fort Lauderdale - 8530, New Zealand','Maia Parker','103','12');
INSERT INTO customer VALUES (DEFAULT,'Ms. Janelle Adams, Black Swan 2573, Lancaster - 5278, Saint Kitts and Nevis','Janelle Adams','113','7');
INSERT INTO customer VALUES (DEFAULT,'Mr. Nate Martin, Clissold 6899, Jersey City - 3532, Suriname','Nate Martin','152','16');
INSERT INTO customer VALUES (DEFAULT,'Ms. Mona Goldsmith, Lake 2715, Laredo - 2426, Japan','Mona Goldsmith','168','16');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Diane Knight, Carolina 1360, Otawa - 2230, Swaziland','Diane Knight','27','12');
INSERT INTO customer VALUES (DEFAULT,'Miss Bree Pitt, Carlisle 6491, Glendale - 2657, Mongolia','Bree Pitt','180','23');
INSERT INTO customer VALUES (DEFAULT,'Ms. Adalind Stanley, Sundown 6558, Otawa - 7556, Andorra','Adalind Stanley','176','16');
INSERT INTO customer VALUES (DEFAULT,'Mr. Phillip Collingwood, Glenwood 9515, Quebec - 8323, Egypt','Phillip Collingwood','133','18');
INSERT INTO customer VALUES (DEFAULT,'Mr. Maxwell Ellery, Andrews 4858, Lyon - 8634, Lesotho','Maxwell Ellery','47','25');
INSERT INTO customer VALUES (DEFAULT,'Mr. Phillip Lee, Betton 1895, Nashville - 2108, Papua New Guinea','Phillip Lee','25','19');
INSERT INTO customer VALUES (DEFAULT,'Miss Sylvia Robinson, Cockspur 9036, Bucharest - 6748, Solomon Islands','Sylvia Robinson','2','24');
INSERT INTO customer VALUES (DEFAULT,'Ms. Rosa Farrell, Durham 8171, Chicago - 5138, India','Rosa Farrell','119','7');
INSERT INTO customer VALUES (DEFAULT,'Ms. Ada Tennant, Chambers 1631, Huntsville - 0158, Sri Lanka','Ada Tennant','3','13');
INSERT INTO customer VALUES (DEFAULT,'Miss Megan Wright, Virgil 7828, Berna - 8611, Austria','Megan Wright','105','25');
INSERT INTO customer VALUES (DEFAULT,'Mr. Barry Gardner, Sheffield 3493, Zurich - 4344, Romania','Barry Gardner','8','21');
INSERT INTO customer VALUES (DEFAULT,'Miss Norah Stone, Collins 9333, Escondido - 7731, Serbia','Norah Stone','24','10');
INSERT INTO customer VALUES (DEFAULT,'Mr. Julius Rothwell, Elia 333, San Francisco - 1134, Sierra Leone','Julius Rothwell','192','7');
INSERT INTO customer VALUES (DEFAULT,'Mr. Harry Whitehouse, Linden 4509, Anaheim - 3681, Rwanda','Harry Whitehouse','79','7');
INSERT INTO customer VALUES (DEFAULT,'Mr. Julian Anderson, South 2549, Quebec - 2027, Ethiopia','Julian Anderson','52','22');
INSERT INTO customer VALUES (DEFAULT,'Mr. Ronald Blackwall, Blean 2894, Salt Lake City - 2381, Sri Lanka','Ronald Blackwall','69','22');
INSERT INTO customer VALUES (DEFAULT,'Mr. Jack Roman, Chandos 8116, Anaheim - 5562, Nigeria','Jack Roman','40','22');
INSERT INTO customer VALUES (DEFAULT,'Miss Trisha Quinn, Buttonwood 1584, Houston - 2757, Korea, South','Trisha Quinn','6','19');
INSERT INTO customer VALUES (DEFAULT,'Mr. Leroy Bennett, Thomas Doyle 7920, Laredo - 0422, Sudan','Leroy Bennett','14','14');
INSERT INTO customer VALUES (DEFAULT,'Mr. Ryan Knight, Bacon 2303, St. Louis - 2625, Israel','Ryan Knight','46','6');
INSERT INTO customer VALUES (DEFAULT,'Ms. Daphne Noach, Clavell 8563, Salem - 7524, Chad','Daphne Noach','171','20');
INSERT INTO customer VALUES (DEFAULT,'Miss Lucy Powell, Carnegie 2638, Oakland - 3282, Malaysia','Lucy Powell','59','12');
INSERT INTO customer VALUES (DEFAULT,'Mr. George Vernon, Emily 5264, Columbus - 5554, Poland','George Vernon','150','6');
INSERT INTO customer VALUES (DEFAULT,'Miss Ellen Thomson, Rivervalley 4874, Baltimore - 3085, Papua New Guinea','Ellen Thomson','137','23');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Meredith Stevens, Parkgate 2339, Escondido - 0824, France','Meredith Stevens','19','25');
INSERT INTO customer VALUES (DEFAULT,'Mr. Mason Flanders, Longman 8488, Fullerton - 8588, Costa Rica','Mason Flanders','97','7');
INSERT INTO customer VALUES (DEFAULT,'Mr. Matt Jones, Eaglet 9014, Pittsburgh - 6820, Sierra Leone','Matt Jones','11','15');
INSERT INTO customer VALUES (DEFAULT,'Mrs. Shay Flack, Collins 4349, Hayward - 4414, Serbia','Shay Flack','140','6');
INSERT INTO customer VALUES (DEFAULT,'Mr. Danny Burnley, Bond 6139, San Bernardino - 4500, Turkey','Danny Burnley','123','17');
INSERT INTO customer VALUES (DEFAULT,'Miss Camellia Windsor, Viscount 8047, Oakland - 8562, Tunisia','Camellia Windsor','199','3');

INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-01-02','YYYY-MM-DD'),TO_DATE('2018-01-24','YYYY-MM-DD'),'130','11','5');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-05-28','YYYY-MM-DD'),TO_DATE('2016-06-03','YYYY-MM-DD'),'83','64','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-02-02','YYYY-MM-DD'),TO_DATE('2019-02-20','YYYY-MM-DD'),'256','11','8');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-11-20','YYYY-MM-DD'),TO_DATE('2021-12-10','YYYY-MM-DD'),'184','36','21');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-06-28','YYYY-MM-DD'),TO_DATE('2018-07-18','YYYY-MM-DD'),'234','6','1');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-10-24','YYYY-MM-DD'),TO_DATE('2020-11-02','YYYY-MM-DD'),'257','86','4');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-05-18','YYYY-MM-DD'),TO_DATE('2018-05-26','YYYY-MM-DD'),'51','92','2');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-09-21','YYYY-MM-DD'),TO_DATE('2016-09-29','YYYY-MM-DD'),'65','33','8');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-09-04','YYYY-MM-DD'),TO_DATE('2017-09-07','YYYY-MM-DD'),'88','75','20');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2022-01-01','YYYY-MM-DD'),TO_DATE('2022-01-07','YYYY-MM-DD'),'56','16','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-11-15','YYYY-MM-DD'),TO_DATE('2019-11-23','YYYY-MM-DD'),'233','53','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-06-03','YYYY-MM-DD'),TO_DATE('2020-06-22','YYYY-MM-DD'),'289','69','11');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-07-27','YYYY-MM-DD'),TO_DATE('2018-08-01','YYYY-MM-DD'),'202','95','5');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-04-16','YYYY-MM-DD'),TO_DATE('2020-04-27','YYYY-MM-DD'),'128','86','19');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-10-19','YYYY-MM-DD'),TO_DATE('2020-11-04','YYYY-MM-DD'),'131','29','5');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-07-06','YYYY-MM-DD'),TO_DATE('2017-07-12','YYYY-MM-DD'),'178','29','9');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-07-02','YYYY-MM-DD'),TO_DATE('2019-07-23','YYYY-MM-DD'),'181','33','5');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-03-30','YYYY-MM-DD'),TO_DATE('2019-04-11','YYYY-MM-DD'),'218','40','24');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-03-08','YYYY-MM-DD'),TO_DATE('2020-03-26','YYYY-MM-DD'),'139','75','1');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-10-07','YYYY-MM-DD'),TO_DATE('2020-10-17','YYYY-MM-DD'),'200','16','21');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-02-06','YYYY-MM-DD'),TO_DATE('2016-02-16','YYYY-MM-DD'),'55','75','12');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-12-27','YYYY-MM-DD'),TO_DATE('2018-12-30','YYYY-MM-DD'),'268','41','24');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-08-20','YYYY-MM-DD'),TO_DATE('2016-08-23','YYYY-MM-DD'),'125','55','19');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-01-23','YYYY-MM-DD'),TO_DATE('2018-01-31','YYYY-MM-DD'),'272','6','6');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-04-29','YYYY-MM-DD'),TO_DATE('2017-05-03','YYYY-MM-DD'),'208','86','5');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-10-17','YYYY-MM-DD'),TO_DATE('2017-10-26','YYYY-MM-DD'),'132','22','9');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-03-25','YYYY-MM-DD'),TO_DATE('2020-04-03','YYYY-MM-DD'),'91','16','22');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-12-10','YYYY-MM-DD'),TO_DATE('2019-12-26','YYYY-MM-DD'),'58','41','11');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-03-11','YYYY-MM-DD'),TO_DATE('2019-03-25','YYYY-MM-DD'),'77','55','21');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-02-17','YYYY-MM-DD'),TO_DATE('2021-03-03','YYYY-MM-DD'),'120','45','23');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-07-29','YYYY-MM-DD'),TO_DATE('2020-08-02','YYYY-MM-DD'),'169','99','21');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2022-05-16','YYYY-MM-DD'),TO_DATE('2022-06-02','YYYY-MM-DD'),'146','60','21');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-08-30','YYYY-MM-DD'),TO_DATE('2016-09-01','YYYY-MM-DD'),'257','6','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-10-20','YYYY-MM-DD'),TO_DATE('2020-10-30','YYYY-MM-DD'),'77','22','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-10-28','YYYY-MM-DD'),TO_DATE('2019-11-09','YYYY-MM-DD'),'298','45','18');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-04-11','YYYY-MM-DD'),TO_DATE('2019-04-18','YYYY-MM-DD'),'217','84','24');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-07-17','YYYY-MM-DD'),TO_DATE('2019-07-23','YYYY-MM-DD'),'160','69','25');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-10-01','YYYY-MM-DD'),TO_DATE('2017-10-22','YYYY-MM-DD'),'123','64','15');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-10-18','YYYY-MM-DD'),TO_DATE('2020-11-06','YYYY-MM-DD'),'151','95','12');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-03-08','YYYY-MM-DD'),TO_DATE('2018-03-22','YYYY-MM-DD'),'244','53','2');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-02-01','YYYY-MM-DD'),TO_DATE('2017-02-05','YYYY-MM-DD'),'116','86','9');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-08-18','YYYY-MM-DD'),TO_DATE('2016-08-24','YYYY-MM-DD'),'225','55','10');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-05-17','YYYY-MM-DD'),TO_DATE('2019-05-25','YYYY-MM-DD'),'264','69','1');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-08-08','YYYY-MM-DD'),TO_DATE('2021-08-28','YYYY-MM-DD'),'289','29','21');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-04-15','YYYY-MM-DD'),TO_DATE('2020-05-01','YYYY-MM-DD'),'236','86','16');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-04-20','YYYY-MM-DD'),TO_DATE('2020-05-01','YYYY-MM-DD'),'272','64','25');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-07-07','YYYY-MM-DD'),TO_DATE('2019-07-24','YYYY-MM-DD'),'264','29','5');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-05-07','YYYY-MM-DD'),TO_DATE('2018-05-11','YYYY-MM-DD'),'187','45','15');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-01-19','YYYY-MM-DD'),TO_DATE('2021-01-28','YYYY-MM-DD'),'276','1','21');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-09-09','YYYY-MM-DD'),TO_DATE('2019-09-27','YYYY-MM-DD'),'222','36','13');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-11-13','YYYY-MM-DD'),TO_DATE('2018-11-19','YYYY-MM-DD'),'223','16','19');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-08-20','YYYY-MM-DD'),TO_DATE('2020-09-02','YYYY-MM-DD'),'71','41','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-12-04','YYYY-MM-DD'),TO_DATE('2020-12-22','YYYY-MM-DD'),'192','22','6');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-12-17','YYYY-MM-DD'),TO_DATE('2018-01-01','YYYY-MM-DD'),'195','84','9');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-11-13','YYYY-MM-DD'),TO_DATE('2017-11-17','YYYY-MM-DD'),'257','14','25');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-11-14','YYYY-MM-DD'),TO_DATE('2018-11-16','YYYY-MM-DD'),'204','69','18');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-03-15','YYYY-MM-DD'),TO_DATE('2017-03-24','YYYY-MM-DD'),'129','22','20');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-01-24','YYYY-MM-DD'),TO_DATE('2019-02-02','YYYY-MM-DD'),'104','22','20');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-06-12','YYYY-MM-DD'),TO_DATE('2020-06-18','YYYY-MM-DD'),'91','75','25');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-05-28','YYYY-MM-DD'),TO_DATE('2019-06-02','YYYY-MM-DD'),'79','60','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2022-05-22','YYYY-MM-DD'),TO_DATE('2022-06-11','YYYY-MM-DD'),'281','1','5');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-04-07','YYYY-MM-DD'),TO_DATE('2021-04-11','YYYY-MM-DD'),'85','6','11');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-02-02','YYYY-MM-DD'),TO_DATE('2020-02-23','YYYY-MM-DD'),'124','16','12');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-11-10','YYYY-MM-DD'),TO_DATE('2018-11-27','YYYY-MM-DD'),'111','22','21');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-06-30','YYYY-MM-DD'),TO_DATE('2018-07-18','YYYY-MM-DD'),'51','69','20');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-05-14','YYYY-MM-DD'),TO_DATE('2021-05-27','YYYY-MM-DD'),'263','45','18');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2022-04-24','YYYY-MM-DD'),TO_DATE('2022-04-27','YYYY-MM-DD'),'67','64','4');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-05-03','YYYY-MM-DD'),TO_DATE('2018-05-19','YYYY-MM-DD'),'284','5','20');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-02-27','YYYY-MM-DD'),TO_DATE('2019-03-19','YYYY-MM-DD'),'103','35','1');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-06-09','YYYY-MM-DD'),TO_DATE('2017-06-29','YYYY-MM-DD'),'104','60','9');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-02-01','YYYY-MM-DD'),TO_DATE('2016-02-16','YYYY-MM-DD'),'233','45','12');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-01-09','YYYY-MM-DD'),TO_DATE('2021-01-14','YYYY-MM-DD'),'78','75','11');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-09-01','YYYY-MM-DD'),TO_DATE('2021-09-09','YYYY-MM-DD'),'71','6','11');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-01-30','YYYY-MM-DD'),TO_DATE('2017-02-16','YYYY-MM-DD'),'291','14','18');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-04-28','YYYY-MM-DD'),TO_DATE('2019-05-14','YYYY-MM-DD'),'72','95','21');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-08-16','YYYY-MM-DD'),TO_DATE('2019-08-18','YYYY-MM-DD'),'152','1','1');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-10-01','YYYY-MM-DD'),TO_DATE('2018-10-11','YYYY-MM-DD'),'53','45','19');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-06-06','YYYY-MM-DD'),TO_DATE('2016-06-09','YYYY-MM-DD'),'250','75','6');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-08-27','YYYY-MM-DD'),TO_DATE('2020-09-08','YYYY-MM-DD'),'251','64','17');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-09-04','YYYY-MM-DD'),TO_DATE('2017-09-17','YYYY-MM-DD'),'198','11','18');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-04-17','YYYY-MM-DD'),TO_DATE('2017-05-07','YYYY-MM-DD'),'196','41','1');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-01-15','YYYY-MM-DD'),TO_DATE('2021-01-28','YYYY-MM-DD'),'54','75','22');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-11-14','YYYY-MM-DD'),TO_DATE('2020-11-25','YYYY-MM-DD'),'264','75','6');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-10-07','YYYY-MM-DD'),TO_DATE('2017-10-10','YYYY-MM-DD'),'221','22','6');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-07-22','YYYY-MM-DD'),TO_DATE('2019-08-03','YYYY-MM-DD'),'165','16','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-03-17','YYYY-MM-DD'),TO_DATE('2016-03-23','YYYY-MM-DD'),'178','55','16');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2020-11-07','YYYY-MM-DD'),TO_DATE('2020-11-13','YYYY-MM-DD'),'268','99','2');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-02-18','YYYY-MM-DD'),TO_DATE('2021-03-02','YYYY-MM-DD'),'227','36','18');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2018-01-12','YYYY-MM-DD'),TO_DATE('2018-01-29','YYYY-MM-DD'),'234','69','6');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-07-13','YYYY-MM-DD'),TO_DATE('2019-07-22','YYYY-MM-DD'),'178','86','13');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-11-15','YYYY-MM-DD'),TO_DATE('2019-12-01','YYYY-MM-DD'),'192','45','2');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2019-09-07','YYYY-MM-DD'),TO_DATE('2019-09-16','YYYY-MM-DD'),'169','45','7');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-08-30','YYYY-MM-DD'),TO_DATE('2016-09-17','YYYY-MM-DD'),'143','95','15');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-06-13','YYYY-MM-DD'),TO_DATE('2017-07-03','YYYY-MM-DD'),'93','1','1');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-04-29','YYYY-MM-DD'),TO_DATE('2017-05-03','YYYY-MM-DD'),'82','75','7');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2016-04-25','YYYY-MM-DD'),TO_DATE('2016-04-29','YYYY-MM-DD'),'147','55','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2022-05-17','YYYY-MM-DD'),TO_DATE('2022-05-27','YYYY-MM-DD'),'286','92','3');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-01-08','YYYY-MM-DD'),TO_DATE('2017-01-12','YYYY-MM-DD'),'166','92','17');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2017-12-19','YYYY-MM-DD'),TO_DATE('2018-01-01','YYYY-MM-DD'),'222','11','22');
INSERT INTO rental VALUES (DEFAULT,TO_DATE('2021-12-15','YYYY-MM-DD'),TO_DATE('2022-01-03','YYYY-MM-DD'),'231','60','13');

INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2019-06-22','YYYY-MM-DD'),'33','5');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2019-03-23','YYYY-MM-DD'),'40','24');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2020-02-22','YYYY-MM-DD'),'75','1');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2020-10-05','YYYY-MM-DD'),'16','21');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2016-02-04','YYYY-MM-DD'),'75','12');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2018-12-11','YYYY-MM-DD'),'41','24');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2016-08-12','YYYY-MM-DD'),'55','19');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2018-01-16','YYYY-MM-DD'),'6','6');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2017-04-13','YYYY-MM-DD'),'86','5');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2017-09-29','YYYY-MM-DD'),'22','9');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2020-03-07','YYYY-MM-DD'),'16','22');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2019-12-03','YYYY-MM-DD'),'41','11');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2019-03-02','YYYY-MM-DD'),'55','21');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2021-02-15','YYYY-MM-DD'),'45','23');
INSERT INTO reservation VALUES (DEFAULT,TO_DATE('2020-07-24','YYYY-MM-DD'),'99','21');

INSERT INTO charge VALUES (DEFAULT,'469.01','2860','179.8','748.79','1','1');
INSERT INTO charge VALUES (DEFAULT,'397.61','498','875.31','198.29','1','2');
INSERT INTO charge VALUES (DEFAULT,'207.94','4608','149.32','593.33','0','3');
INSERT INTO charge VALUES (DEFAULT,'414.95','3680','549.34','30.96','0','4');
INSERT INTO charge VALUES (DEFAULT,'308.85','4680','151.24','578.64','0','5');
INSERT INTO charge VALUES (DEFAULT,'192.32','2313','926.83','260.46','1','6');
INSERT INTO charge VALUES (DEFAULT,'141.24','408','255.78','879.58','1','7');
INSERT INTO charge VALUES (DEFAULT,'373.01','520','516.39','74.89','1','8');
INSERT INTO charge VALUES (DEFAULT,'9.62','264','338.89','301.61','1','9');
INSERT INTO charge VALUES (DEFAULT,'119.61','336','965.23','640.15','1','10');
INSERT INTO charge VALUES (DEFAULT,'419.04','1864','820.98','243.23','0','11');
INSERT INTO charge VALUES (DEFAULT,'91.64','5491','618.36','485.75','0','12');
INSERT INTO charge VALUES (DEFAULT,'119.13','1010','687.35','979.25','0','13');
INSERT INTO charge VALUES (DEFAULT,'177.42','1408','513.93','125.9','0','14');
INSERT INTO charge VALUES (DEFAULT,'6.34','2096','490.94','836.56','0','15');
INSERT INTO charge VALUES (DEFAULT,'11.51','1068','761.65','453.56','1','16');
INSERT INTO charge VALUES (DEFAULT,'173.88','3801','454.57','203.99','0','17');
INSERT INTO charge VALUES (DEFAULT,'136.36','2616','122.3','20.48','1','18');
INSERT INTO charge VALUES (DEFAULT,'222.2','2502','224.27','809.57','1','19');
INSERT INTO charge VALUES (DEFAULT,'295.37','2000','136.44','420.69','1','20');
INSERT INTO charge VALUES (DEFAULT,'405.59','550','485.43','675.76','0','21');
INSERT INTO charge VALUES (DEFAULT,'241.63','804','680.36','373.68','1','22');
INSERT INTO charge VALUES (DEFAULT,'107.04','375','825.84','568.44','0','23');
INSERT INTO charge VALUES (DEFAULT,'430.44','2176','900.88','208.28','0','24');
INSERT INTO charge VALUES (DEFAULT,'234.09','832','528.13','359.37','0','25');
INSERT INTO charge VALUES (DEFAULT,'331.41','1188','95.2','681.24','0','26');
INSERT INTO charge VALUES (DEFAULT,'179.41','819','591.52','372.77','1','27');
INSERT INTO charge VALUES (DEFAULT,'242','928','957.47','33.31','1','28');
INSERT INTO charge VALUES (DEFAULT,'425.21','1078','238.9','661.8','1','29');
INSERT INTO charge VALUES (DEFAULT,'34.57','1680','363.53','948.5','0','30');
INSERT INTO charge VALUES (DEFAULT,'409.58','676','559.64','14.15','0','31');
INSERT INTO charge VALUES (DEFAULT,'197.36','2482','366.22','402.91','0','32');
INSERT INTO charge VALUES (DEFAULT,'53.06','514','197.3','902','1','33');
INSERT INTO charge VALUES (DEFAULT,'287.7','770','634.21','983.34','0','34');
INSERT INTO charge VALUES (DEFAULT,'36.58','3576','119.35','229.84','1','35');
INSERT INTO charge VALUES (DEFAULT,'57.59','1519','201.9','42.11','1','36');
INSERT INTO charge VALUES (DEFAULT,'368.95','960','337.08','584.57','0','37');
INSERT INTO charge VALUES (DEFAULT,'315.98','2583','70.52','404.61','1','38');
INSERT INTO charge VALUES (DEFAULT,'107.81','2869','887.58','322.67','1','39');
INSERT INTO charge VALUES (DEFAULT,'190.86','3416','81.52','335.98','1','40');
INSERT INTO charge VALUES (DEFAULT,'346.1','464','687.3','634.74','1','41');
INSERT INTO charge VALUES (DEFAULT,'455.48','1350','193.96','131.14','1','42');
INSERT INTO charge VALUES (DEFAULT,'325.95','2112','833.5','181.9','1','43');
INSERT INTO charge VALUES (DEFAULT,'195.64','5780','163.52','248.72','0','44');
INSERT INTO charge VALUES (DEFAULT,'38.66','3776','213.42','803.34','0','45');
INSERT INTO charge VALUES (DEFAULT,'255.39','2992','919.63','50.54','1','46');
INSERT INTO charge VALUES (DEFAULT,'446.38','4488','602.89','529.5','0','47');
INSERT INTO charge VALUES (DEFAULT,'394.73','748','151.62','320.06','0','48');
INSERT INTO charge VALUES (DEFAULT,'466.56','2484','850.2','908.64','1','49');
INSERT INTO charge VALUES (DEFAULT,'187.88','3996','678','157.92','0','50');
INSERT INTO charge VALUES (DEFAULT,'80.48','1338','874.99','752.88','1','51');
INSERT INTO charge VALUES (DEFAULT,'251.51','923','723.69','133.02','1','52');
INSERT INTO charge VALUES (DEFAULT,'13.27','3456','526.1','446.46','0','53');
INSERT INTO charge VALUES (DEFAULT,'10.64','2925','858.46','702.05','0','54');
INSERT INTO charge VALUES (DEFAULT,'324.81','1028','370.37','126.95','1','55');
INSERT INTO charge VALUES (DEFAULT,'329','408','850.36','597.56','0','56');
INSERT INTO charge VALUES (DEFAULT,'302.9','1161','93.15','369.92','1','57');
INSERT INTO charge VALUES (DEFAULT,'372.51','936','307.24','779.25','1','58');
INSERT INTO charge VALUES (DEFAULT,'154.35','546','820.64','577.37','0','59');
INSERT INTO charge VALUES (DEFAULT,'59.14','395','667.91','500.83','1','60');
INSERT INTO charge VALUES (DEFAULT,'204.89','5620','596.7','517.43','0','61');
INSERT INTO charge VALUES (DEFAULT,'126.06','340','765.68','375.31','1','62');
INSERT INTO charge VALUES (DEFAULT,'489.22','2604','809.78','391.66','1','63');
INSERT INTO charge VALUES (DEFAULT,'314.67','1887','926.71','37.66','0','64');
INSERT INTO charge VALUES (DEFAULT,'0.1','918','246.45','250.33','0','65');
INSERT INTO charge VALUES (DEFAULT,'12.79','3419','944.2','71.47','0','66');
INSERT INTO charge VALUES (DEFAULT,'268.65','201','901.39','445.95','0','67');
INSERT INTO charge VALUES (DEFAULT,'79.53','4544','548.96','453.7','0','68');
INSERT INTO charge VALUES (DEFAULT,'330.07','2060','782.78','350.83','0','69');
INSERT INTO charge VALUES (DEFAULT,'416.39','2080','393.36','502.32','0','70');
INSERT INTO charge VALUES (DEFAULT,'243.1','3495','877.45','582.43','1','71');
INSERT INTO charge VALUES (DEFAULT,'394.68','390','498.59','508.86','0','72');
INSERT INTO charge VALUES (DEFAULT,'343.35','568','672.12','126.49','0','73');
INSERT INTO charge VALUES (DEFAULT,'237.24','4947','455.24','182.18','1','74');
INSERT INTO charge VALUES (DEFAULT,'26.55','1152','231.74','746.46','0','75');
INSERT INTO charge VALUES (DEFAULT,'216','304','991.29','200.3','0','76');
INSERT INTO charge VALUES (DEFAULT,'446.6','530','710.16','969.1','0','77');
INSERT INTO charge VALUES (DEFAULT,'127.53','750','356.98','500.73','1','78');
INSERT INTO charge VALUES (DEFAULT,'234.06','3012','284.26','371.85','0','79');
INSERT INTO charge VALUES (DEFAULT,'358.37','2574','54.01','71.6','0','80');
INSERT INTO charge VALUES (DEFAULT,'266.86','3920','163.81','324.11','0','81');
INSERT INTO charge VALUES (DEFAULT,'340.59','702','942.61','529.78','1','82');
INSERT INTO charge VALUES (DEFAULT,'266.97','2904','413.42','289.27','1','83');
INSERT INTO charge VALUES (DEFAULT,'315.81','663','581.22','478.97','1','84');
INSERT INTO charge VALUES (DEFAULT,'331.33','1980','540.63','140.62','0','85');
INSERT INTO charge VALUES (DEFAULT,'290.01','1068','630.2','779.91','0','86');
INSERT INTO charge VALUES (DEFAULT,'497.94','1608','938.38','124.87','0','87');
INSERT INTO charge VALUES (DEFAULT,'441.48','2724','328.87','385.54','0','88');
INSERT INTO charge VALUES (DEFAULT,'75.78','3978','438.77','392.8','0','89');
INSERT INTO charge VALUES (DEFAULT,'84.95','1602','928.64','199.16','0','90');
INSERT INTO charge VALUES (DEFAULT,'250.84','3072','126.72','754.99','1','91');
INSERT INTO charge VALUES (DEFAULT,'243.01','1521','737.34','610.89','0','92');
INSERT INTO charge VALUES (DEFAULT,'99.35','2574','700.94','190.18','0','93');
INSERT INTO charge VALUES (DEFAULT,'306.24','1860','251.28','515.47','0','94');
INSERT INTO charge VALUES (DEFAULT,'67.14','328','651','887.65','0','95');
INSERT INTO charge VALUES (DEFAULT,'399.26','588','81.16','976.13','1','96');
INSERT INTO charge VALUES (DEFAULT,'93.96','2860','272.77','327.29','0','97');
INSERT INTO charge VALUES (DEFAULT,'114.66','664','823.71','490.46','0','98');
INSERT INTO charge VALUES (DEFAULT,'299.33','2886','572.7','252.72','0','99');
INSERT INTO charge VALUES (DEFAULT,'32.22','4389','205.8','453.53','1','100');
