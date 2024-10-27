import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;

public class Util {
    static byte[] concatBytes(byte[] l, byte[] r) {
        byte[] concat = new byte[l.length + r.length];
        for (int i = 0; i < concat.length; ++i)
            concat[i] = i < l.length ? l[i] : r[i - l.length];
        return concat;
    }

    static byte[] serialize(Object obj) {
        byte[] ser = null;
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream out = null;
        try {
            out = new ObjectOutputStream(bos);
            out.writeObject(obj);
            out.flush();
            ser = bos.toByteArray();
        }  catch (IOException e) {
            System.out.println(e);
            System.out.println("Problem serializing");
        } finally {
            try {
                bos.close();
            } catch (IOException e) {
                System.out.println(e);
                System.out.println("Problem closing");
            }
        }
        return ser;
    }
}
