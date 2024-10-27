-- 1
    SELECT id FROM (
        SELECT id, COUNT(*) FROM (
            SELECT id, course_id, COUNT(*) AS num FROM takes
            GROUP BY id, course_id
            HAVING COUNT(*) >= 2
            ORDER BY id
        )
        GROUP BY id
        HAVING COUNT(*) > 2
    )
    ORDER BY CAST(id AS INTEGER);

-- 2
    select name, id from instructor i
    where not exists (
        select c.dept_name from course c
        where c.dept_name = i.dept_name
        and not exists (
            select t.id from teaches t
            where t.id = i.id
            and t.course_id = c.course_id
        )
    )
    order by name;

-- 3
    select s.name, s.id from student s
    where s.dept_name='Cybernetics' and
        s.name like 'F%' and
        ((select count(*) from takes t
        join course c on c.course_id = t.course_id
        where t.id=s.id and c.dept_name='Music') -
        (select count(*) from course where dept_name='Music')) < 4;

-- 4
    select avg(salary) from instructor;

    select sum(salary)/count(*) from instructor;

   -- They are not equal becuase there is a null salary instructor table which means that the avg method will be bigger than the second method because avg will ignore the null value wheras sum does not.

-- 5
   -- a
        select id from instructor
        minus
        select id from teaches;
   -- b
        select id from instructor
        natural left join teaches
        where course_id is null;
   -- c
        select id from instructor
        where id not in (select id from teaches);

-- 6
    -- a)
        -- a) Would be different because of the minus operator
        -- b) would be the same because it employs a left outer join
        -- c) would be different
    -- b)
       -- each field in the teaches table are included in the primary key and primay key fields may not be null

-- 7
    create table physicscourse (
        course_id varchar(7),
        title varchar(50),
        dept_name varchar(20),
        credits numeric(2,0) check (credits > 0),
        primary key (course_id));

    create table physicsinstructor (
        id varchar (5),
        name varchar (20) not null,
        dept_name varchar (20),
        salary numeric (8,2) check (salary > 29000),
        primary key (id));

    create table physicsteaches (
        id varchar (5),
        course_id varchar (8),
        sec_id varchar (8),
        semester varchar (6),
        year numeric (4,0),
        primary key (id, course_id, sec_id, semester, year));

    -- ^^^ note that I was unable to include the foreign keys to tables that I do not have permission on.

    grant select on physicscourse to grader;
    grant select on physicsinstructor to grader;
    grant select on physicsteaches to grader;

    insert into physicsinstructor
    select * from instructor
    where dept_name = 'Physics';

    insert into physicscourse
    select * from course
    where dept_name = 'Physics';

    insert into physicsteaches
    select * from teaches
    where id in (
        select id from physicsinstructor
    ) and
    course_id in (
        select course_id from physicscourse
    );

    insert into physicsinstructor values(2, 'Duo', 'Physics', 29001);

    insert into physicsteaches (
        select 2, course_id, 1, 'Spring', 2016 from physicscourse
        where course_id >= 200 and course_id < 300);
