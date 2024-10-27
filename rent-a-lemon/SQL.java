import java.sql.*;

class SQL {
    static PreparedStatement searchCustomer(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT customer_id, name " +
            "FROM customer " +
            "WHERE name LIKE ? " +
            "ORDER BY name"
        );
    }

    static PreparedStatement selectCustomer(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT customer_id " +
            "FROM customer " +
            "WHERE customer_id = ?"
        );
    }

    static PreparedStatement createLicense(Connection c) throws SQLException {
        return c.prepareStatement(
            "INSERT INTO license (license_id, license_number, state, age) " +
            "VALUES (DEFAULT, ?, ?, ?)",
            new String[] { "license_id" }
        );
    }

    static PreparedStatement createCustomer(Connection c) throws SQLException {
        return c.prepareStatement(
            "INSERT INTO customer (address, name, license_id, membership_id) " +
            "VALUES (?, ?, ?, ?)",
            new String[] { "customer_id" }
        );
    }

    static PreparedStatement searchMembership(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT membership_id, group_name " +
            "FROM membership " +
            "WHERE group_name LIKE ?" +
            "ORDER BY group_name"
        );
    }

    static PreparedStatement getMembershipByID(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT membership_id, group_name " +
            "FROM membership " +
            "WHERE membership_id = ?"
        );
    }

    static PreparedStatement searchLocation(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT location_id, street, city, state, zip " +
            "FROM location " +
            "WHERE city LIKE ?" +
            "ORDER BY city"
        );
    }

    static PreparedStatement selectLocation(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT location_id " +
            "FROM location " +
            "WHERE location_id = ?"
        );
    }

    static PreparedStatement listVehicles(Connection c) throws SQLException {
        return c.prepareStatement(
            "select v.vehicle_id, v.make, v.model, v.year, v.color, v.type " +
            "from vehicle v " +
            "where v.location_id = ? " +
            "and v.vehicle_id not in " +
                "(select vehicle_id from vehicle " +
                "natural join reservation " +
                "where created between sysdate - 10 and sysdate) " +
            "and v.vehicle_id not in " +
                "(select vehicle_id from vehicle " +
                "natural join rental " +
                "where returned is null) " +
            "order by v.vehicle_id",
            ResultSet.TYPE_SCROLL_INSENSITIVE,
            ResultSet.CONCUR_UPDATABLE
        );
    }

    static PreparedStatement selectVehicle(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT vehicle_id " +
            "FROM vehicle " +
            "WHERE vehicle_id = ?"
        );
    }

    static PreparedStatement createReservation(Connection c) throws SQLException {
        return c.prepareStatement(
            "INSERT INTO reservation (created, customer_id, vehicle_id) " +
            "VALUES (sysdate, ?, ?)"
        );
    }

    static PreparedStatement deleteReservation(Connection c) throws SQLException {
        return c.prepareStatement(
            "DELETE FROM reservation " +
            "WHERE reservation_id = ? AND customer_id = ?"
        );
    }

    static PreparedStatement listReservations(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT reservation_id, vehicle_id, location_id, city, created, make, model, year, color, type " +
            "FROM reservation NATURAL JOIN vehicle NATURAL JOIN location " +
            "WHERE customer_id = ? " +
            "ORDER BY created"
        );
    }

    static PreparedStatement listReservationsAtLocation(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT vehicle_id, reservation_id, location_id, city, created, make, model, year, color, type " +
            "FROM reservation NATURAL JOIN vehicle NATURAL JOIN location " +
            "WHERE customer_id = ? " +
            "AND location_id = ? " +
            "AND created between sysdate - 10 and sysdate " +
            "ORDER BY created",
            ResultSet.TYPE_SCROLL_INSENSITIVE,
            ResultSet.CONCUR_UPDATABLE
        );
    }

    static PreparedStatement completeCharge(Connection c) throws SQLException {
        return c.prepareStatement(
            "UPDATE charge " +
            "SET complete=1 " +
            "where charge_id in ( " +
            "    select charge_id from charge " +
            "    natural join rental " +
            "    natural join customer " +
            "    where customer_id = ? " +
            "       and complete = 0 " +
            "       and charge_id = ? " +
            ")"
        );
    }

    static PreparedStatement listRentals (Connection c) throws SQLException {
        return c.prepareStatement(
            "select rental_id, taken, returned, rate, model, year, color, type, city, " +
            "state from rental " +
            "natural join customer " +
            "natural join vehicle " +
            "natural join location " +
            "where customer_id = ? " +
            "order by taken"
        );
    }

    static PreparedStatement listCharges(Connection c) throws SQLException {
        return c.prepareStatement(
            "select charge_id, fuel, dropoff, insurance, other, rate, percent_off, " +
            "    round(((fuel + dropoff + insurance + other + rate) * (1-(percent_off / 100))),2) " +
            "    as total from charge " +
            "natural join rental " +
            "natural join customer " +
            "natural join membership " +
            "natural join discount " +
            "where customer_id = ? " +
            "order by charge_id"
        );
    }

    static PreparedStatement createRental(Connection c) throws SQLException {
        return c.prepareStatement(
            "INSERT INTO rental (taken, returned, rate, customer_id, vehicle_id) " +
            "VALUES (sysdate, null, ?, ?, ?)"
        );
    }

    static PreparedStatement listNonreturnedRentals (Connection c) throws SQLException {
        return c.prepareStatement(
            "select rental_id, taken, returned, rate, make, model, year, color, type, city, " +
            "state from rental " +
            "natural join customer " +
            "natural join vehicle " +
            "natural join location " +
            "where customer_id = ? " +
            "and returned is null " +
            "order by taken"
        );
    }

    static PreparedStatement acceptReturn(Connection c) throws SQLException {
        return c.prepareStatement(
            "UPDATE rental " +
            "SET returned = sysdate " +
            "where rental_id in ( " +
            "    select rental_id from rental " +
            "    natural join customer " +
            "    where customer_id = ? " +
            "    and returned is null " +
            "    and rental_id = ? " +
            ")"
        );
    }

    static PreparedStatement createCharge(Connection c) throws SQLException {
        return c.prepareStatement(
            "INSERT INTO charge (fuel, dropoff, insurance, other, complete, rental_id) " +
            "VALUES (?, ?, ?, ?, 0, ?)"
        );
    }

    static PreparedStatement listVehicleCounts(Connection c) throws SQLException {
        return c.prepareStatement(
            "select t1.location_id, total, NVL(taken, 0) as taken, NVL(total - taken, total) as remaining from " +
            "(select count(location_id) total, location_id from vehicle natural join location group by location_id) t1 " +
            "left outer join (select count(v.location_id) taken, v.location_id " +
            "    from vehicle v " +
            "    where v.vehicle_id in " +
            "        (select vehicle_id from vehicle " +
            "        natural join reservation " +
            "        where created between sysdate - 10 and sysdate) " +
            "    or v.vehicle_id in " +
            "        (select vehicle_id from vehicle " +
            "        natural join rental " +
            "        where returned is null) " +
            "    group by v.location_id " +
            ") t2 on t1.location_id = t2.location_id " +
            "order by remaining"
        );
    }

    static PreparedStatement createVehicle(Connection c) throws SQLException {
        return c.prepareStatement(
            "INSERT INTO vehicle (make, model, year, color, type, vin, odo, location_id) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            new String[] { "vehicle_id" }
        );
    }

    static PreparedStatement moveVehicle(Connection c) throws SQLException {
        return c.prepareStatement(
            "update vehicle " +
            "set location_id = ? " +
            "where location_id in ( " +
            "    select location_id from vehicle " +
            "    where vehicle_id = ? " +
            "    and location_id = ? " +
            ") and vehicle_id = ?"
        );
    }

    static PreparedStatement createLocation(Connection c) throws SQLException {
        return c.prepareStatement(
            "INSERT INTO location (street, city, state, zip) " +
            "VALUES (?, ?, ?, ?)",
            new String[] { "location_id" }
        );
    }

    static PreparedStatement listOutstandingChargesForCustomer(Connection c) throws SQLException {
        return c.prepareStatement(
            "select charge_id, location_id, fuel, dropoff, insurance, other, " +
            "    rate, round(returned - taken, 4) as num_days, percent_off, " +
            "    round(((fuel + dropoff + insurance + other + (rate * (returned - taken))) * (1 - (percent_off / 100))),2) " +
            "    as total from charge " +
            "natural join rental " +
            "natural join customer " +
            "natural join membership " +
            "natural join discount " +
            "natural join vehicle " +
            "where customer_id = ? " +
            "    and complete = 0 " +
            "order by location_id"
        );
    }

    static PreparedStatement listAllOutstandingCharges(Connection c) throws SQLException {
        return c.prepareStatement(
            "select customer_id, location_id, charge_id, fuel, dropoff, insurance, other, " +
            "    rate, round(returned - taken, 4) as num_days, percent_off, " +
            "    round(((fuel + dropoff + insurance + other + (rate * (returned - taken))) * (1 - (percent_off / 100))),2) " +
            "    as total from charge " +
            "natural join rental " +
            "natural join customer " +
            "natural join membership " +
            "natural join discount " +
            "natural join vehicle " +
            "where complete = 0 " +
            "order by customer_id"
        );
    }

    static PreparedStatement listOutstandingChargesForLocation(Connection c) throws SQLException {
        return c.prepareStatement(
            "select charge_id, customer_id, fuel, dropoff, insurance, other, " +
            "    rate, round(returned - taken, 4) as num_days, percent_off, " +
            "    round(((fuel + dropoff + insurance + other + (rate * (returned - taken))) * (1 - (percent_off / 100))),2) " +
            "    as total from charge " +
            "natural join rental " +
            "natural join customer " +
            "natural join membership " +
            "natural join discount " +
            "natural join vehicle " +
            "where complete = 0 " +
            "    and location_id = ? " +
            "order by customer_id"
        );
    }

    static PreparedStatement getRental(Connection c) throws SQLException {
        return c.prepareStatement(
            "SELECT rental_id, location_id, round(sysdate - taken, 4) as num_days, vehicle_id " +
            "FROM rental NATURAL JOIN vehicle " +
            "WHERE customer_id = ? " +
            "    AND rental_id = ?"
        );
    }

    static PreparedStatement updateOdometer(Connection c) throws SQLException {
        return c.prepareStatement(
            "UPDATE vehicle " +
            "SET odo = ? " +
            "where vehicle_id in ( " +
            "    select vehicle_id from rental " +
            "    natural join vehicle " +
            "    natural join customer " +
            "    where customer_id = ? " +
            "    and rental_id = ? " +
            "    and vehicle_id = ? " +
            ")"
        );
    }

    static PreparedStatement listVehicleStatuses(Connection c) throws SQLException {
        return c.prepareStatement(
            "select vehicle_id, make, model, year, color, type, vin, odo, " +
            "    case when vehicle_id in ( " +
            "        select vehicle_id " +
            "        from vehicle v " +
            "        where location_id = ? " +
            "        and v.vehicle_id in " +
            "            (select vehicle_id from vehicle " +
            "                natural join reservation " +
            "                where created between sysdate - 10 and sysdate) " +
            "        or v.vehicle_id in " +
            "            (select vehicle_id from vehicle " +
            "            natural join rental " +
            "            where returned is null) " +
            "    ) then 'UNAVAILABLE' " +
            "    else  'AVAILABLE' end as status " +
            "from vehicle " +
            "natural join location " +
            "where location_id = ? " +
            "order by status"
        );
    }
}
