import java.util.ArrayList;

//Struct containing proof of membership data
public class POM {
    String balance;
    ArrayList<byte[]> merkleProof;
    BlockHeader blockHeader;
    ArrayList<byte[]> blockHashes;

    POM(String balance, ArrayList<byte[]> merkleProof, BlockHeader blockHeader, ArrayList<byte[]> blockHashes) {
        this.balance = balance;
        this.merkleProof = merkleProof;
        this.blockHeader = blockHeader;
        this.blockHashes = blockHashes;
    }

    public String toString() {
        String print = "";
        print += "Balance:\n\t" + balance + "\n";
        print += "Block Header:\n" + blockHeader;
        print += "MerkleProof:\n";
        for (byte[] h: this.merkleProof)
            print += "\t" + Hash.hashToHex(h) + "\n";
        print += "blockHashes:\n";
        for (byte[] h: this.blockHashes)
            print += "\t" + Hash.hashToHex(h) + "\n";
        return print;
    }
}
