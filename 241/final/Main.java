import java.util.Scanner;
import java.sql.*;
import java.io.Console;

public class Main {
    Connection c;
    Scanner s;
    Bridge bridge;
    Customer customer;
    Staff staff;
    Manager manager;

    public static void main(String[] args) {
        new Main().run();
    }

    Main() {
        this.s = new Scanner(System.in);
        this.bridge = new Bridge(this);
        this.login();
    }

    void run() {
        System.out.println("Interface list:");
        System.out.println("\t[1] Customer");
        System.out.println("\t[2] Staff");
        System.out.println("\t[3] Manager");
        System.out.println("\t[X] Exit");
        Bridge.prompt("Select an interface > ");
        switch (s.nextLine()) {
            case "1":
                if (this.customer == null) this.customer = new Customer(this);
                this.customer.run();
                break;
            case "2":
                if (this.staff == null) this.staff = new Staff(this);
                this.staff.run();
                break;
            case "3":
                if (this.manager == null) this.manager = new Manager(this);
                this.manager.run();
                break;
            case "X":
            case "x":
                System.out.println("Gracefully exiting program.");
                this.s.close();
                try {
                    this.c.close();
                    System.exit(0);
                } catch (Exception e) {
                    System.exit(0);
                }
                break;
            default:
                Bridge.errln("Invalid option, try again.");
                run();
        }
    }

    void login() {
        String login = this.bridge.getString("Input your edgar1 oracle login > ", 64);
        if (login == null) {
            Bridge.errln("You may not exit out of this prompt. Try again.");
            this.login();
        }
        Console console = System.console();
        Bridge.prompt("Input your edgar 1 oracle password > ");
        String pwd = new String(console.readPassword());

        Connection c;
        try {
            c = DriverManager.getConnection("jdbc:oracle:thin:@edgar1.cse.lehigh.edu:1521:cse241", login, pwd);
            c.setAutoCommit(false);
            this.c = c;
            System.out.println("Welcome.");
        } catch (SQLException e) {
            Bridge.errln("Invalid credentials, try again.");
            login();
        }
    }
}
