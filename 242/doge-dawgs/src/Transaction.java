import java.nio.charset.StandardCharsets;

class Transaction {
    String address;
    String balance;
    byte[] hash;

    Transaction(String address, String balance) {
        this.address = address;
        this.balance = balance;
        this.hash = Hash.hash((address + balance).getBytes(StandardCharsets.UTF_8));
    }
}
