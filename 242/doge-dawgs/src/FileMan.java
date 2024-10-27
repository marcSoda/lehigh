import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.ListIterator;
import java.io.IOException;
import java.math.BigInteger;

public class FileMan {
    static Transaction[][] readLedgerFiles(String paths) throws Exception {
        ArrayList<File> files = new ArrayList<File>();
        String[] split = paths.split(" ");
        for (String path : split) {
            try {
                File file = new File(path);
                if (!file.exists()) {
                    throw new Exception("Error: " + path + " does not exist");
                }
                if (file.isDirectory()) {
                    for (File f : file.listFiles()) {
                        files.add(f);
                    }
                } else {
                    files.add(file);
                }
            } catch (Exception e) {
                throw new Exception("Error: when reading from " + path);
            }
        }
        Transaction[][] transactions = new Transaction[files.size()][];
        int count = 0;
        for (File file : files) {
            try {
                Scanner s;
                try {
                    s = new Scanner(file);
                } catch (Exception e) {
                    throw new Exception("Error: failed to initialize scanner for " + file.getName());
                }

                Transaction[] data = getLedger(s);
                s.close();

                transactions[count] = data;
                count++;
            } catch (Exception e) {
                throw new Exception("Error " + file.getName() + " is an invalid ledger");
            }
        }
        return transactions;
    }

    private static Transaction[] getLedger(Scanner s) {
        ArrayList<String> lines = new ArrayList<String>();
        while (s.hasNextLine()) {
            String l = s.nextLine().trim();
            if (l.equals("END TRANSACTIONS"))
                break;
            lines.add(l);
        }

        lines.sort(String.CASE_INSENSITIVE_ORDER);

        Transaction[] ledger = new Transaction[lines.size()];
        for (int i = 0; i < lines.size(); i++) {
            String[] split = lines.get(i).split(" ");
            Transaction t = new Transaction(split[0], split[1]);
            ledger[i] = t;
        }
        return ledger;
    }

    static String getFirstInputName(String paths) throws Exception {
        String[] split = paths.split(" ");
        for (String path : split) {
            try {
                File file = new File(path);
                if (!file.exists()) {
                    throw new Exception("Error: " + path + " does not exist");
                }
                if (file.isDirectory()) {
                    for (File f : file.listFiles()) {
                        return f.getName();
                    }
                } else {
                    return file.getName();
                }
            } catch (Exception e) {
                throw new Exception("Error: when reading from " + path);
            }
        }
        throw new Exception("Unable to find filename");
    }

    static void writeBlockchain(Blockchain bc, String fname) throws Exception {
        int firstIdx = 0;
        if (fname.contains("/"))
            firstIdx = fname.lastIndexOf('/') + 1;
        int lastIdx = fname.length();
        if (fname.contains("."))
            lastIdx = fname.lastIndexOf('.');
        String outName = fname.substring(firstIdx, lastIdx) + ".block.out";
        ListIterator<Block> it = bc.descendingIterator();
        try {
            File outFile = new File(outName);
            outFile.delete();
            outFile.createNewFile();
            FileWriter w = new FileWriter(outFile);
            while (it.hasPrevious()) {
                w.write(it.previous().toString(true));
            }
            w.close();
        } catch (IOException e) {
            throw new Exception("Error: IOException when writing blockchain");
        }
    }

    static Blockchain readBlockchain(String path) throws Exception {
        ArrayList<Block> bl = new ArrayList<Block>();
        try {
            Scanner s = new Scanner(new File(path));
            while (s.hasNextLine()) {
                bl.add(readBlock(s));
            }
            s.close();
        } catch (Exception e) {
            throw new Exception("Failed to read blockchain file.");
        }
        return new Blockchain(bl);
    }

    private static Block readBlock(Scanner s) throws Exception {
        String l = s.nextLine().trim();
        if (!l.equals("BEGIN BLOCK")) {
            throw new Exception("Error: Bad block: Could not find beginning of block.");
        }
        while (s.hasNextLine()) {
            l = s.nextLine().trim();
            if (l.equals("END BLOCK"))
                break;
            BlockHeader head = new BlockHeader();
            while (s.hasNextLine()) {
                l = s.nextLine().trim();
                if (l.equals("END HEADER"))
                    break;
                String[] split = l.split("\\s+");

                switch (split[0]) {
                    case "PreviousHash:":
                        head.prev = Hash.hexToHash(split[1]);
                        break;
                    case "MerkleRootHash:":
                        head.rootHash = Hash.hexToHash(split[1]);
                        break;
                    case "Timestamp:":
                        head.timestamp = split[1];
                        break;
                    case "Target:":
                        head.target = new BigInteger(split[1]);
                        break;
                    case "Nonce:":
                        head.nonce = Long.parseLong(split[1]);
                        break;
                    default:
                        throw new Exception("Error when parsing block header.");
                }
            }
            if (!s.nextLine().trim().equals("BEGIN TRANSACTIONS")) {
                throw new Exception("Error: Bad Block: Could not find beginning of ledger.");
            }
            Transaction[] ledger = getLedger(s);
            if (!s.nextLine().trim().equals("END BLOCK")) {
                throw new Exception("Error: Bad Block: Could not find end of block.");
            }

            return new Block(head, ledger);
        }
        return null;
    }
}
