import java.util.ArrayList;
import java.io.Serializable;

class Node implements Serializable {
    byte[] hash;
    Node left;
    Node right;

    Node(byte[] hash, Node left, Node right) {
        this.hash = hash;
        this.left = left;
        this.right = right;
    }
}

public class MerkleTree implements Serializable {
    Node root;

    MerkleTree(Transaction[] transactions) {
        if (transactions.length == 0) {
            System.out.println("Fatal Error: A block must have at least one transaction.");
            System.exit(1);
        }

        ArrayList<Node> leaves = new ArrayList<Node>();

        for (Transaction t : transactions) {
            leaves.add(new Node(t.hash, null, null));
        }

        if (leaves.size() % 2 != 0) {
            Node last = leaves.get(leaves.size() - 1);
            leaves.add(new Node(last.hash, null, null));
        }

        this.root = this.buildTree(leaves);
    }

    Node buildTree(ArrayList<Node> nodes) {
        if (nodes.size() % 2 != 0) {
            Node last = nodes.get(nodes.size() - 1);
            nodes.add(new Node(last.hash, null, null));
        }

        if (nodes.size() == 2) {
            byte[] hash = Hash.hash(Util.concatBytes(nodes.get(0).hash, nodes.get(1).hash));
            return new Node(hash, nodes.get(0), nodes.get(1));
        }

        Node left = this.buildTree(new ArrayList<>(nodes.subList(0, (nodes.size()) / 2)));
        Node right = this.buildTree(new ArrayList<>(nodes.subList((nodes.size()) / 2, nodes.size())));
        byte[] hash = Hash.hash(Util.concatBytes(left.hash, right.hash));
        return new Node(hash, left, right);
    }

    static boolean find(Transaction t, Node node) {
        if (node == null)
            return false;
        if (node.hash.equals(t.hash)) {
            return true;
        }
        return find(t, node.left) || find(t, node.right);
    }

    static ArrayList<byte[]> reqProof(Transaction t, Node root) {
        if (!find(t, root)) {
            System.out.println("Transaction not found");
            return null;
        }

        ArrayList<byte[]> proof = new ArrayList<byte[]>();
        buildProof(root, t, proof);
        proof.add(root.hash);

        return proof;
    }

    static boolean buildProof(Node n, Transaction t, ArrayList<byte[]> proof) {
        if (n == null)
            return false;
        if (n.hash.equals(t.hash))
            return true;

        boolean foundLeft = buildProof(n.left, t, proof);
        boolean foundRight = buildProof(n.right, t, proof);

        if (!foundLeft && !foundRight)
            return false;

        if (foundLeft && foundRight && proof.contains(t.hash))
            return false;

        if (foundLeft) {
            proof.add(n.left.hash);
            proof.add(n.right.hash);
        } else if (foundRight) {
            proof.add(n.right.hash);
            proof.add(n.left.hash);
        }

        return true;
    }

    static void printTree(Node node) {
        if (node != null) {
            System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
            if (node.left != null) {
                System.out.println("LEFT: " + Hash.hashToHex(node.left.hash));
                System.out.println("RIGHT: " + Hash.hashToHex(node.right.hash));
            }
            System.out.println("SELF: " + Hash.hashToHex(node.hash));
            System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
            printTree(node.left);
            printTree(node.right);
        }
    }
}
