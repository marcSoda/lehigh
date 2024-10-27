import java.util.Random;
import java.math.BigInteger;
import java.util.List;

class TestSuite {
    Blockchain bc;
    String path;

    TestSuite(String path) {
        this.path = path;
        try {
            this.bc = FileMan.readBlockchain(this.path);
        } catch (Exception e) {
            System.out.println(e);
            System.exit(1);
        }
        this.reset();
    }

    private void reset() {
        try {
            this.bc = FileMan.readBlockchain(this.path);
        } catch (Exception e) {
            System.out.println(e);
            System.exit(1);
        }
    }

    private Block getRandomBlock() {
        List<Block> chain = bc.getChain();
        return chain.get(new Random().nextInt(chain.size()));
    }

    void testReadBlockchain() {
        System.out.print("TestReadBlockchain:\n" +
                "\tTests the verification of a blockchain that has been read from an outfile\n" +
                "\tExpected: true\n" +
                "\tTest Result: Valid Blockchain: ");
        System.out.println(this.bc.verify());
        this.reset();
    }

    void testManipulateHeaderPreviousHash() {
        System.out.print("TestManipulateHeaderPreviousHash\n" +
                "\tTests the verification of a blockchain where the previous hash of a random block has been manipulated\n" +
                "\tExpected: false\n" +
                "\tTest Result: Valid Blockchain: ");
        Block bad = getRandomBlock();
        bad.header.prev[0] = 17;
        System.out.println(this.bc.verify());
        this.reset();
    }

    void testManipulateHeaderMerkleRootHash() {
        System.out.print("TestManipulateHeaderMerkleRootHash\n" +
                "\tTests the verification of a blockchain where the merkle root hash of a random block has been manipulated\n" +
                "\tExpected: false\n" +
                "\tTest Result: Valid Blockchain: ");
        Block bad = getRandomBlock();
        bad.header.rootHash[0] = 17;
        System.out.println(this.bc.verify());
        this.reset();
    }

    void testManipulateHeaderTimestamp() {
        System.out.print("TestManipulateHeaderTimestamp\n" +
                "\tTests the verification of a blockchain where the timestamp of a random block has been manipulated\n" +
                "\tExpected: false\n" +
                "\tTest Result: Valid Blockchain: ");
        Block bad = getRandomBlock();
        bad.header.timestamp = "Bad timestamp";
        System.out.println(this.bc.verify());
        this.reset();
    }

    void testManipulateHeaderTarget() {
        System.out.print("TestManipulateHeaderTarget\n" +
                "\tTests the verification of a blockchain where the target of a random block has been manipulated\n" +
                "\tExpected: false\n" +
                "\tTest Result: Valid Blockchain: ");
        Block bad = getRandomBlock();
        bad.header.target = BigInteger.ONE;
        System.out.println(this.bc.verify());
        this.reset();
    }

    void testManipulateLedger() {
        System.out.print("TestManipulateLedger\n" +
                "\tTests the verification of a blockchain where a random transaction of the ledger of a random block has been manipulated\n" +
                "\tExpected: false each time\n" +
                "\tThis test is run 10 times. Results:\n");
        for (int i = 0; i < 10; i++) {
            Block bad = getRandomBlock();
            Transaction[] badLedger = bad.transactions;
            int randLedgerIdx = new Random().nextInt(badLedger.length);
            badLedger[randLedgerIdx] = new Transaction("BadAccount", "999999999");
            System.out.println("\tTest " + i + " Result: Valid Blockchain: " + this.bc.verify());
            this.reset();
        }
    }

    void run() {
        System.out.println("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        System.out.println("Running the test suite:");
        System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<");
        this.testReadBlockchain();
        this.testManipulateHeaderPreviousHash();
        this.testManipulateHeaderMerkleRootHash();
        this.testManipulateHeaderTimestamp();
        this.testManipulateHeaderTarget();
        this.testManipulateLedger();
    }
}
