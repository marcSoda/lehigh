import java.util.Scanner;
import java.io.*;
import java.sql.*;

public class TeachingRecords {
    public static void main(String[] args) {
        Scanner stdin = new Scanner(System.in);
        System.out.println("Please input your Oracle username on Edgar1:");
        String login = stdin.nextLine();
        System.out.println("Please input your Oracle password on Edgar1:");
        Console console = System.console();
        char [] pwd = console.readPassword();

        try (Connection con = DriverManager.getConnection("jdbc:oracle:thin:@edgar1.cse.lehigh.edu:1521:cse241", login, new String(pwd));) {
            showInstructorList(stdin, con);
            showTeachingRecord(stdin, con);
        } catch(Exception e) { e.printStackTrace(); }
    }

    static void showInstructorList(Scanner stdin, Connection con) {
        System.out.print("\nInput name search substring: ");
        String sub = stdin.nextLine();
        if (sub.contains("'")) {
            System.out.println("Name search substring may not contain '");
            showInstructorList(stdin, con);
            return;
        }

        try (Statement statement = con.createStatement();) {
            ResultSet result;
            String q = "SELECT id, name FROM instructor " +
                        "WHERE name LIKE '%" + sub + "%'";
            result = statement.executeQuery(q);
            if (!result.next()) {
                System.out.println ("Empty result.");
                showInstructorList(stdin, con);
                return;
            } else {
                System.out.println("\nHere is a list of matching IDs:\n");
                String fmtStr = "%-7s%-40s";
                System.out.println(String.format(fmtStr, "ID", "Name"));
                do {
                    System.out.println(String.format(fmtStr, result.getString(1), result.getString(2)));
                } while (result.next());
            }
        } catch(Exception e) {
            System.out.println("0: An error has occurred: try again.");
            showInstructorList(stdin, con);
            return;
        }
    }

    static void showTeachingRecord(Scanner stdin, Connection con) {
        try (Statement statement = con.createStatement();) {
            System.out.print("\nInput instructor ID: ");
            String tid = stdin.nextLine();
            if (tid.contains("'")) {
                System.out.println("Name search substring may not contain a single quote. Try again.");
                showTeachingRecord(stdin, con);
                return;
            } else if (!tid.matches("[0-9]+") || Integer.valueOf(tid) < 0 || Integer.valueOf(tid) > 99999) {
                System.out.println("Enter an integer between 0 and 99999.");
                showTeachingRecord(stdin, con);
                return;
            }

            String ins = instructorExists(tid, stdin, con);
            if (ins == null) {
                System.out.println("Could not find an instructor with id " + tid);
                showTeachingRecord(stdin, con);
                return;
            }

            ResultSet result;
            String q = "select c.dept_name, c.course_id, c.title, te.sec_id, te.semester, te.year, ta.enrollment from instructor i " +
                        "inner join teaches te on te.id = i.id " +
                        "inner join ( " +
                        "    select count(*) as enrollment, course_id, semester, year from takes group by course_id, semester, year " +
                        ") ta on ta.course_id = te.course_id and ta.semester=te.semester and ta.year=te.year " +
                        "inner join course c on c.course_id=te.course_id " +
                        "where i.id = " + tid + " " +
                        "order by c.dept_name, c.course_id, te.year, te.semester";
            result = statement.executeQuery(q);
            if (!result.next()) {
                System.out.println ("Instructor does not have a teaching record. Try again.");
                showTeachingRecord(stdin, con);
                return;
            } else {
                System.out.println("\nShowing teaching record for instructor: " + ins + " with ID: " + tid + "\n");
                String fmtStr = "%-12s%-5s%-50s%-4s%-8s%-6s%-11s";
                System.out.println(String.format(fmtStr, "Department", "CNO", "Title", "Sec", "Sem", "Year", "Enrollment"));
                do {
                    System.out.println(String.format(fmtStr, result.getString(1), result.getString(2), result.getString(3), result.getString(4), result.getString(5), result.getString(6), result.getString(7)));
                } while (result.next());
            }
        } catch(Exception e) {
            System.out.println("1: An error has occurred. Try again.");
            showTeachingRecord(stdin, con);
            return;
        }
    }

    static String instructorExists(String tid, Scanner stdin, Connection con) {
        try (Statement statement = con.createStatement();) {
            ResultSet result;
            String q = "SELECT name FROM instructor WHERE id=" + tid + "";
            result = statement.executeQuery(q);
            if (!result.next()) {
                return null;
            } else {
                return result.getString(1);
            }
        } catch(Exception e) {
            System.out.println("2: An error has occurred. Try again.");
            return null;
        }
    }
}
