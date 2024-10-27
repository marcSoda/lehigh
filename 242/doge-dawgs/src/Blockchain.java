import java.math.BigInteger;
import java.util.LinkedList;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;

public class Blockchain {
    private byte[] lastHash = firstHash();

    private List<Block> chain = new LinkedList<Block>();

    Blockchain() {
        Block genesis = defaultGenesis();
        this.add(genesis);
    }

    //Only used for reading blockchains from an output file.
    Blockchain(ArrayList<Block> blockList) {
        for (int i = blockList.size()-1; i >= 0; i--) {
            this.chain.add(blockList.get(i));
        }
    }

    public void add(Block block) {
        block.header.prev = lastHash;
        lastHash = block.mine();
        chain.add(block);
    }

    public boolean verify() {
        Iterator<Block> it = this.iterator();
        byte[] prev = firstHash();
        while (it.hasNext()) {
            Block b = it.next();
            if (!Arrays.equals(b.header.prev, prev))
                return false;
            if (!b.verify())
                return false;
            prev = Hash.hash(Util.serialize(b.header));
        }
        return true;
    }

    public POM balance(String address) {
        ListIterator<Block> dit = this.descendingIterator();
        Block foundBlock = null;
        Transaction foundTransaction = null;

        outer:
        while (dit.hasPrevious()) {
            Block b = dit.previous();
            for (Transaction t : b.transactions) {
                if (address.equals(t.address)) {
                    foundBlock = b;
                    foundTransaction = t;
                    break outer;
                }
            }
        }

        if (foundTransaction == null || foundBlock == null) {
            System.out.println("Could not find matching transaction.");
            return null;
        }

        ArrayList<byte[]> merkleProof = MerkleTree.reqProof(foundTransaction, foundBlock.root);

        ArrayList<byte[]> blockHashes = new ArrayList<byte[]>();
        while (dit.hasNext()) {
            blockHashes.add(Hash.hash(Util.serialize(dit.next().header)));
        }

        return new POM(
                   foundTransaction.balance,
                   merkleProof,
                   foundBlock.header,
                   blockHashes
        );
    }

    public String toString(boolean showLedger) {
        String print = "";
        for (Block block : this.chain) {
            print += block.toString(showLedger);
        }
        return print;
    }

    static byte[] firstHash() {
        return BigInteger.valueOf(0).toByteArray();
    }

    static Block defaultGenesis() {
        Transaction genesisTransaction = new Transaction("Genesis", "0");
        Transaction[] transactions = { genesisTransaction };
        return new Block(transactions);
    }

    public List<Block> getChain() {
        return this.chain;
    }

    public Iterator<Block> iterator() {
        return chain.iterator();
    }

    public ListIterator<Block> descendingIterator() {
        return this.chain.listIterator(this.chain.size());
    }
}
