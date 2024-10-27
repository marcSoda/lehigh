import java.util.ArrayList;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.sql.*;

class Bridge {
    Main main;

    Bridge(Main main) {
        this.main = main;
    }

    //returns -1 to go back to customer menu
    int selectCustomer() {
        if (this.binaryQuery("Would you like to search for your customer ID?"))
            if (!this.searchCustomer()) return -1;
        int cid = this.getInt("Input your customer ID > ");
        if (cid == -1) return -1;
        try {
            PreparedStatement ps = SQL.selectCustomer(this.main.c);
            ps.setInt(1, cid);
            ResultSet r = ps.executeQuery();
            if (!r.isBeforeFirst()) {
                errln("There are no customers matching that ID.");
                return this.selectCustomer();
            } else {
                r.next();
                int resCid = r.getInt(1);
                System.out.println("Customer with ID of " + resCid + " selected.\n");
                return resCid;
            }
        } catch(Exception e) {
            defaultErr();
            return this.selectCustomer();
        }
    }

    //return false when x hit
    boolean searchCustomer() {
        String sub = this.getString("Input your name or a substring of your name > ", 256);
        if (sub == null) return false;
        try {
            PreparedStatement ps = SQL.searchCustomer(this.main.c);
            ps.setString(1, "%" + sub + "%");
            ResultSet r = ps.executeQuery();
            if (!r.isBeforeFirst()) {
                errln("There are no customers matching that name.");
                return this.searchCustomer();
            } else {
                System.out.println("\nHere is a list of matching customers:\n");
                Printer.print(r);
                return true;
            }
        } catch(Exception e) {
            defaultErr();
            return this.searchCustomer();
        }
    }

    //used by customer and clerk
    //returns -1 when X hit, there are no vehicles, or lid is -1
    int selectVehicle(int lid) {
        ArrayList<Integer> vids = this.listVehicles(lid);
        if (vids == null) return -1; //if vids is null, then there are no vehicles at the selected location
        int vid = this.getInt("Input the vehicle id > ");
        if (vid == -1) return -1;
        //ensure that the selected vehicle is in the returned list of vehicles at the selected location
        if (!vids.contains(vid)) {
            errln("The vehicle that you have selected is not at the location that you selected.");
            return this.selectVehicle(lid);
        }
        try {
            PreparedStatement ps = SQL.selectVehicle(this.main.c);
            ps.setInt(1, vid);
            ResultSet r = ps.executeQuery();
            if (!r.isBeforeFirst()) {
                errln("There are no vehicles matching that ID.");
                return this.selectVehicle(lid);
            }
            r.next();
            int resInt = r.getInt(1);
            System.out.println("Vehicle with ID of " + resInt + " selected.\n");
            return resInt;
        } catch(Exception e) {
            defaultErr();
            return this.selectVehicle(lid);
        }
    }

    //used by customer and clerk
    ArrayList<Integer> listVehicles(int lid) {
        try {
            PreparedStatement ps = SQL.listVehicles(this.main.c);
            ps.setInt(1, lid);
            ResultSet r = ps.executeQuery();
            ArrayList<Integer> vids = new ArrayList<Integer>();
            if (!r.isBeforeFirst()) {
                errln("There are no vehicles at this location.");
                return null;
            } else {
                System.out.println("\nHere is a list of available vehicles at that location:\n");
                Printer.print(r);
                r.beforeFirst();
                while (r.next()) vids.add(r.getInt(1));
                return vids;
            }
        } catch(Exception e) {
            defaultErr();
            return null;
        }
    }

    //returns -1 if X was hit
    int selectLocation() {
        if (this.binaryQuery("Would you like to search for your location?"))
            if (!this.searchLocation()) return -1;
        int lid = this.getInt("Input your location ID > ");
        if (lid == -1) return -1;
        try {
            PreparedStatement ps = SQL.selectLocation(this.main.c);
            ps.setInt(1, lid);
            ResultSet r = ps.executeQuery();
            if (!r.isBeforeFirst()) {
                errln("There are no locations matching that ID.");
                return this.selectLocation();
            }
            r.next();
            int resLid = r.getInt(1);
            System.out.println("Location with ID of " + resLid + " selected.\n");
            return resLid;
        } catch(Exception e) {
            defaultErr();
            return this.selectLocation();
        }
    }

    //returns false when X hit
    boolean searchLocation() {
        String sub = this.getString("Search for your location city > ", 128);
        if (sub == null) return false;
        try {
            PreparedStatement ps = SQL.searchLocation(this.main.c);
            ps.setString(1, "%" + sub + "%");
            ResultSet r = ps.executeQuery();
            if (!r.isBeforeFirst()) {
                errln("There are no locations matching that name.");
                return this.searchLocation();
            } else {
                System.out.println("\nHere is a list of matching locations:\n");
                Printer.print(r);
                return true;
            }
        } catch(Exception e) {
            defaultErr();
            return this.searchLocation();
        }
    }

    //returns false when there are no outstanding charges or error
    boolean listOutstandingChargesForCustomer(int cid) {
        try {
            PreparedStatement ps = SQL.listOutstandingChargesForCustomer(this.main.c);
            ps.setInt(1, cid);
            ResultSet r = ps.executeQuery();
            if (!r.isBeforeFirst()) {
                Bridge.errln("There are no outstanding charges.");
                return false;
            } else {
                System.out.println("\nHere is a list of outstanding charges:\n");
                Printer.print(r);
                return true;
            }
        } catch(Exception e) {
            Bridge.defaultErr();
            return false;
        }
    }

    void listOutstandingChargesForLocation(int lid) {
        try {
            PreparedStatement ps = SQL.listOutstandingChargesForLocation(this.main.c);
            ps.setInt(1, lid);
            ResultSet r = ps.executeQuery();
            if (!r.isBeforeFirst()) {
                Bridge.errln("No outstanding charges.");
                return;
            } else {
                System.out.println("\nHere is a list of outstanding charges:\n");
                Printer.print(r);
                return;
            }
        } catch(Exception e) {
            Bridge.defaultErr();
            return;
        }
    }

    //return false on error
    boolean moveVehicle(int fromLid, int toLid, int vid, boolean commit) {
        try {
            PreparedStatement ps = SQL.moveVehicle(this.main.c);
            ps.setInt(1, toLid);
            ps.setInt(2, vid);
            ps.setInt(3, fromLid);
            ps.setInt(4, vid);
            if (ps.executeUpdate() == 0) {
                Bridge.defaultErr();
                Bridge.errln("You may have selected a vehicle that does not belong to the location you specified.");
                return false;
            }
            if (commit) this.main.c.commit();
            System.out.println("You have moved the vehicle of ID " + vid + " to the location of ID " + toLid);
            return true;
        } catch(Exception e) {
            Bridge.defaultErr();
            return false;
        }
    }

    //returns false on error
    boolean deleteReservation(int rid, int cid, boolean commit) {
        try {
            PreparedStatement ps = SQL.deleteReservation(this.main.c);
            ps.setInt(1, rid);
            ps.setInt(2, cid);
            if (ps.executeUpdate() == 0) {
                Bridge.defaultErr();
                Bridge.errln("You may have entered a reservation ID that does not belong to you.");
                return this.deleteReservation(rid, cid, commit);
            }
            if (commit) this.main.c.commit();
            System.out.println("You have deleted the reservation of ID " + rid);
            return true;
        } catch(Exception e) {
            Bridge.defaultErr();
            return this.deleteReservation(rid, cid, commit);
        }
    }

    void rollback() {
        try {
            this.main.c.rollback();
            Bridge.errln("All pending changes have been reverted.");
        } catch (Exception e) {
            Bridge.errln("A fatal error has occurred.\nGracefully exiting.\nPlease try again.");
            System.exit(0);
        }
    }

    static String ansiGreen = "\u001B[32m";
    static String ansiRed = "\u001B[31m";
    static String ansiBlue = "\u001B[34m";
    static String ansiRst = "\u001B[0m";

    static void prompt(String prompt) {
        System.out.print(ansiGreen + prompt + ansiRst);
    }

    static void promptln(String prompt) {
        System.out.println(ansiGreen + prompt + ansiRst);
    }

    static void errln(String err) {
        System.out.println(ansiRed + err + ansiRst);
    }

    static void headingln(String heading) {
        System.out.println(ansiBlue + heading + ansiRst);
    }

    static void defaultErr() {
        errln("An error has occurred. Try again");
    }

    static double round(Double val) {
        BigDecimal bd = BigDecimal.valueOf(val);
        bd = bd.setScale(2, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    static Double stod(String val) {
        try { return Double.parseDouble(val); }
        catch (Exception e) {
            errln("Failed to parse input");
            return -1.;
        }
    }

    static int stoi(String val) {
        try { return Integer.parseInt(val); }
        catch (Exception e) {
            errln("Failed to parse input");
            return -1;
        }
    }

    static String itos(int val) {
        try { return Integer.toString(val); }
        catch (Exception e) {
            errln("Failed to parse input");
            return null;
        }
    }

    boolean binaryQuery(String prompt) {
        prompt(prompt + " Y/N > ");
        String resp = this.main.s.nextLine();
        if (resp.equalsIgnoreCase("Y")) return true;
        else if (resp.equalsIgnoreCase("N")) return false;
        else {
            errln("Invalid response.");
            return this.binaryQuery(prompt);
        }
    }

    Double getDouble(String query) {
        prompt(query);
        String line = this.main.s.nextLine();
        if (line.equalsIgnoreCase("X")) return null;
        try { return round(Double.parseDouble(line)); }
        catch (NumberFormatException e) {
            errln("Input must be a decimal");
            return this.getDouble(query);
        }
    }

    int getInt(String query) {
        prompt(query);
        String line = this.main.s.nextLine();
        if (line.equalsIgnoreCase("X")) return -1;
        try { return Integer.parseUnsignedInt(line); }
        catch (NumberFormatException e) {
            errln("Input must be a non-negative integer");
            return this.getInt(query);
        }
    }

    String getString(String query, int maxLen) {
        prompt(query);
        String line = this.main.s.nextLine();
        if (line.equalsIgnoreCase("X")) return null;
        if (line.length() == 0) {
            errln("Input must not be empty.");
            return this.getString(query, maxLen);
        } else if (line.contains("'")) {
            errln("Input may not contain single quotes.");
            return this.getString(query, maxLen);
        } else if (line.length() > maxLen) {
            errln("Input may not exceed " + maxLen + " characters.");
            return this.getString(query, maxLen);
        }
        return line;
    }
}
