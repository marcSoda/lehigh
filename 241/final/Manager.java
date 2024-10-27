import java.sql.*;

class Manager {
    Main main;

    Manager(Main main) {
        this.main = main;
    }

    void run() {
        System.out.println("\nManager Interface");
        System.out.println("\t[1] View number of vehicles at each location");
        System.out.println("\t[2] Add a vehicle");
        System.out.println("\t[3] Move a vehicle");
        System.out.println("\t[4] Add a location");
        System.out.println("\t[5] View all unpaid charges");
        System.out.println("\t[6] View all unpaid charges for a customer");
        System.out.println("\t[7] View all unpaid charges for a location");
        System.out.println("\t[X] Return to main menu");
        Bridge.prompt("What would you like to do? > ");
        switch (this.main.s.nextLine()) {
            case "1":
                this.listVehicleCounts();
                break;
            case "2":
                this.createVehicle();
                break;
            case "3":
                this.moveVehicle();
                break;
            case "4":
                this.createLocation();
                break;
            case "5":
                this.listAllOutstandingCharges();
                break;
            case "6":
                this.listOutstandingChargesForCustomer();
                break;
            case "7":
                this.listOutstandingChargesForLocation();
                break;
            case "X":
            case "x":
                this.main.run();
                return;
            default:
                Bridge.errln("Invalid option. Try again.");
        }
        this.run();
    }

    void listVehicleCounts() {
        try {
            PreparedStatement ps = SQL.listVehicleCounts(this.main.c);
            ResultSet r = ps.executeQuery();
            if (!r.isBeforeFirst()) {
                Bridge.errln("No information is available");
                this.run();
            } else {
                System.out.println("\nHere are the vehicle counts by location:\n");
                Printer.print(r);
            }
        } catch(Exception e) {
            Bridge.errln("An error has occurred.");
            this.run();
        }
    }

    void createVehicle() {
        int lid = this.main.bridge.selectLocation();
        String make = this.main.bridge.getString("Input the make > ", 64);
        if (make == null) return;
        String model = this.main.bridge.getString("Input the model > ", 64);
        if (model == null) return;
        int year = this.main.bridge.getInt("Input the year > ");
        if (year == -1) return;
        String color = this.main.bridge.getString("Input the color > ", 16);
        if (color == null) return;
        String type = this.main.bridge.getString("Input the type > ", 16);
        if (type == null) return;
        String vin = this.main.bridge.getString("Input the VIN > ", 17);
        if (vin == null) return;
        int odo = this.main.bridge.getInt("Input the Odometer Reading > ");
        if (odo == -1) return;
        try {
            PreparedStatement ps = SQL.createVehicle(this.main.c);
            ps.setString(1, make);
            ps.setString(2, model);
            ps.setInt(3, year);
            ps.setString(4, color);
            ps.setString(5, type);
            ps.setString(6, vin);
            ps.setInt(7, odo);
            ps.setInt(8, lid);

            if (ps.executeUpdate() == 0) {
                Bridge.defaultErr();
                this.createVehicle();
            }
            ResultSet rs = ps.getGeneratedKeys();
            rs.next();
            int vid = rs.getInt(1);
            System.out.println(
                "\nVehicle creation summary:" +
                "\n\tMake:            " + make +
                "\n\tModel:           " + model +
                "\n\tYear:            " + year +
                "\n\tColor:           " + color +
                "\n\tType:            " + type +
                "\n\tVin:             " + vin +
                "\n\tOdometer:        " + odo +
                "\n\tLocation ID:     " + lid
            );
            if (!this.main.bridge.binaryQuery("Are the above details correct?"))
                return;
            this.main.c.commit();
            System.out.println("Vehicle with ID " + vid + " created.");
            this.run();
        } catch(Exception e) {
            Bridge.defaultErr();
            this.createVehicle();
        }
    }

    void moveVehicle() {
        Bridge.promptln("Selecting the location to take a vehicle from:");
        int fromLid = this.main.bridge.selectLocation();
        if (fromLid == -1) return;
        int vid = this.main.bridge.selectVehicle(fromLid);
        if (vid == -1) return;
        Bridge.promptln("Selecting the location to move the vehicle to:");
        int toLid = this.main.bridge.selectLocation();
        if (toLid == -1) return;
        if (!this.main.bridge.moveVehicle(fromLid, toLid, vid, true)) this.moveVehicle();
    }

    void createLocation() {
        String street = this.main.bridge.getString("Input the Street > ", 128);
        if (street == null) return;
        String city = this.main.bridge.getString("Input the City > ", 128);
        if (city == null) return;
        String state = this.main.bridge.getString("Input the State > ", 64);
        if (state == null) return;
        int zip = this.main.bridge.getInt("Input the Zip Code > ");
        if (zip == -1) return;
        try {
            PreparedStatement ps = SQL.createLocation(this.main.c);
            ps.setString(1, street);
            ps.setString(2, city);
            ps.setString(3, state);
            ps.setInt(4, zip);

            if (ps.executeUpdate() == 0) {
                Bridge.defaultErr();
                this.createLocation();
            }
            ResultSet rs = ps.getGeneratedKeys();
            rs.next();
            int lid = rs.getInt(1);
            System.out.println(
                "\nLocation creation summary:" +
                "\n\tStreet:            " + street +
                "\n\tCity:              " + city +
                "\n\tState:             " + state +
                "\n\tZip Code:          " + zip
            );
            if (!this.main.bridge.binaryQuery("Are the above details correct?"))
                return;
            this.main.c.commit();
            System.out.println("Location with ID " + lid + " created.");
            this.run();
        } catch(Exception e) {
            Bridge.defaultErr();
            this.createLocation();
        }
    }

    void listAllOutstandingCharges() {
        try {
            PreparedStatement ps = SQL.listAllOutstandingCharges(this.main.c);
            ResultSet r = ps.executeQuery();
            if (!r.isBeforeFirst()) {
                Bridge.errln("There are no outstanding charges.");
                this.run();
            } else {
                System.out.println("\nHere is a list of all outstanding charges:\n");
                Printer.print(r);
            }
        } catch(Exception e) {
            Bridge.defaultErr();
        }
    }

    void listOutstandingChargesForCustomer() {
        int cid = this.main.bridge.selectCustomer();
        if (cid == -1) return;
        this.main.bridge.listOutstandingChargesForCustomer(cid);
    }

    void listOutstandingChargesForLocation() {
        int lid = this.main.bridge.selectLocation();
        if (lid == -1) return;
        this.main.bridge.listOutstandingChargesForLocation(lid);
    }
}
