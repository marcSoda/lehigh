//Please note that this class was modified from https://github.com/htorun/dbtableprinter
import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;

public class Printer {
    private static class Column {
        private String label;
        private int width = 0;
        private List<String> values = new ArrayList<>();
        public Column (String label, int type, String typeName) {
            this.label = label;
        }
    }

    public static void print(ResultSet rs) throws Exception {
        ResultSetMetaData rsmd;
        rsmd = rs.getMetaData();
        int columnCount = rsmd.getColumnCount();
        List<Column> columns = new ArrayList<>(columnCount);
        List<String> tableNames = new ArrayList<>(columnCount);

        for (int i = 1; i <= columnCount; i++) {
            Column c = new Column(rsmd.getColumnLabel(i),
                    rsmd.getColumnType(i), rsmd.getColumnTypeName(i));
            c.width = c.label.length();
            columns.add(c);
            if (!tableNames.contains(rsmd.getTableName(i))) {
                tableNames.add(rsmd.getTableName(i));
            }
        }
        int rowCount = 0;
        while (rs.next()) {
            for (int i = 0; i < columnCount; i++) {
                Column c = columns.get(i);
                String value;
                value = rs.getString(i+1) == null ? "NULL" : rs.getString(i+1);
                c.width = value.length() > c.width ? value.length() : c.width;
                c.values.add(value);
            }
            rowCount++;
        }
        StringBuilder strToPrint = new StringBuilder();
        StringBuilder rowSeparator = new StringBuilder();
        for (Column c : columns) {
            String toPrint;
            String name = c.label;
            int diff = c.width - name.length();
            if ((diff%2) == 1) {
                c.width++;
                diff++;
            }
            int paddingSize = diff/2;
            String padding = new String(new char[paddingSize]).replace("\0", " ");
            toPrint = "| " + padding + name + padding + " ";
            strToPrint.append(toPrint);
        }
        String lineSeparator = System.getProperty("line.separator");
        lineSeparator = lineSeparator == null ? "\n" : lineSeparator;
        strToPrint.append("|").append(lineSeparator);
        strToPrint.insert(0, rowSeparator);
        strToPrint.append(rowSeparator);
        StringJoiner sj = new StringJoiner(", ");
        for (String name : tableNames) {
            sj.add(name);
        }
        System.out.print(Bridge.ansiBlue + strToPrint.toString() + Bridge.ansiRst);
        String format;
        for (int i = 0; i < rowCount; i++) {
            for (Column c : columns) {
                format = String.format("| %%%s%ds ", "-", c.width);
                System.out.print(
                        String.format(format, c.values.get(i))
                );
            }
            System.out.println("|");
            System.out.print(rowSeparator);
        }
        System.out.println();
    }
}
