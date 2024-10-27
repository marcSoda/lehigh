-- 1
    SELECT ID FROM instructor WHERE dept_name='Math';

-- 2
    SELECT course_id, title, grade
    FROM takes
    NATURAL JOIN course
    WHERE ID = 10101
    ORDER BY course_id;

-- 3
    SELECT DISTINCT s.ID, s.NAME, s.TOT_CRED FROM student s
        INNER JOIN (
            SELECT ID FROM takes t
            INNER JOIN (
                SELECT COURSE_ID FROM course
                WHERE DEPT_NAME='Music'
            ) m ON m.COURSE_ID = t.COURSE_ID
        ) c ON c.ID=s.ID
    WHERE TOT_CRED > 100 AND DEPT_NAME='Mech. Eng.'
    ORDER BY s.TOT_CRED DESC;

-- 4
    SELECT DISTINCT id, name
    FROM student, (SELECT MAX(tot_cred) AS c FROM student) m
    WHERE DEPT_NAME='Statistics' AND tot_cred < m.c/10
    ORDER BY CAST(id as INTEGER) ASC;

-- 5
    SELECT ROUND(AVG(salary)) as AVG_SALARY
    FROM instructor
    WHERE dept_name='Philosophy';

-- 6
    SELECT dept_name, COUNT(*) as total_advised
    FROM instructor i, advisor a
    WHERE i.id = a.i_id
    GROUP BY dept_name
    ORDER BY dept_name;

    --OR

    SELECT i.DEPT_NAME, COUNT(a.s_id) AS total_advised FROM instructor i
    INNER JOIN advisor a ON a.I_ID = i.ID
    GROUP BY i.dept_name
    ORDER BY i.dept_name;

    /*
    I'm not sure which is faster, but I prefer the joining approach.
    The missing dept (mech eng) does not appear in the output because it does not appear in the instructor table.
    This could be solved by joining the department table. The way to do this is not obvious to me because in Oracle, unlike MySQL, you must have an ON clause with all JOINs.
    */

-- 7
    SELECT DISTINCT a.course_id, a.id FROM takes a
    JOIN (
        SELECT id, course_id, COUNT(*) FROM takes
        GROUP BY id, course_id
        HAVING COUNT(*) > 2
    ) b ON a.id = b.id AND a.course_id = b.course_id
    ORDER BY course_id;
