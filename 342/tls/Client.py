import base64
import secrets
from socket import socket, AF_INET, SOCK_STREAM
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

HEADERLEN = 8
HASHLEN = 32
SMALLCHOMPLEN = 8
BIGCHOMPLEN = 16384
# HOST = ''
HOST = 'varda'
PORT = 4041

class Client:
    cnonce = None      # client nonce
    spub = None        # server public key
    snonce = None      # server nonce
    sock = None        # server socket
    cipher = None      # AES encryption cypher

    def __init__(self):
        self.cnonce = secrets.token_bytes(8)
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.connect((HOST, PORT))

    # send based on a protocol that I created
    # first sends hash for serverside malformed data checking
    # then sends a header containing the number of bytes in payload
    # then sends the data
    def send(self, data: bytes):
        header = len(data).to_bytes(HEADERLEN, 'little')
        self.sock.send(header)
        self.sock.send(hashBytes(data))
        self.sock.send(data)

    # recv based on a protocol that I created
    # first receives a hash for local malformed data checking
    # then receives a header containing the number of bytes in payload
    # then receives the data in a buffered fashon
    def recv(self):
        data = bytes()
        header = self.sock.recv(HEADERLEN)
        hash = self.sock.recv(HASHLEN)
        datalen = int.from_bytes(header, 'little')
        chomplen = 0
        if (datalen > BIGCHOMPLEN): chomplen = BIGCHOMPLEN
        else: chomplen = SMALLCHOMPLEN
        while 1:
            d = self.sock.recv(chomplen)
            data += d
            if len(data) >= datalen: break
        if (hash != hashBytes(data)):
            print("Fatal Error: malformed data")
            exit(1)
        return data

    # receives and decrypts data using the symmetric key generated during the handshake
    def get_aes(self):
        decryptor = self.cipher.decryptor()
        return decryptor.update(self.recv()) + decryptor.finalize()

    # encrypts and sends data using symmetric key generated during the handshake
    def send_aes(self, data):
        encryptor = self.cipher.encryptor()
        self.send(encryptor.update(pad(data, 16)) + encryptor.finalize())

    # perform tls handshake
    def handshake(self):
        print("beginning handshake")
        self.hello()
        print("sent client hello")
        self.await_server_hello()
        print("received server hello")
        pms = self.send_premaster()
        print("sent premaster")
        self.constructMaster(pms)
        print("constructed master")
        self.await_done()
        print("received server done message")
        self.send_done()
        print("sent done message")

    # client hello
    # sends the client nonce. later used to generate the premaster secret on both sides
    # normally the client hello would contain TLS versions supported by the client as well as supported cipher suites
    def hello(self):
        self.send(self.cnonce)

    # waits for server hello
    # receives the server's pub key and nonce
    # pub key is used by the to encrypt the premaster secret to be sent back to the server
    # nonce is used to calculate the premaster secret on both sides
    # normally this would contain an SSL cert that the client could verify, but my server does not have an SSL cert
    def await_server_hello(self):
        rsnonce = self.recv()
        rspub = self.recv()
        self.snonce = rsnonce
        self.spub = serialization.load_pem_public_key(rspub, default_backend())

    # generates premaster secret
    # encrupts premaster secret with server's pub key
    # sends encrupted premaster secret with the server. only the server can feasably decrypt it
    # returns premaster secret used to construct the master secret
    def send_premaster(self):
        pms = secrets.token_bytes(16)
        enc_pms = base64.b64encode(self.spub.encrypt(
            pms,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        ))
        self.send(enc_pms)
        return pms

    # using the premaster secret, client nonce, and server nonce, construct the master secret
    # this is the AES symmetric encryption key
    def constructMaster(self, pms):
        key = pms + self.cnonce + self.snonce
        iv = self.cnonce + self.snonce
        self.cipher = Cipher(algorithms.AES(key), modes.CBC(iv))

    # await and decrypt the 'done' message using the generated symmetric encryption key
    # this signifies that the server is ready for secure communication
    # if the 'done' message is unable to be decrypted, the handshake has failed
    def await_done(self):
        dec = self.get_aes()
        if (dec != pad(b"done", 16)):
            print("error in handshake")
            exit(1)

    # encrupt the 'done' message using the symmetric key
    # signifies that the client is ready for secure communication
    # if the server is unable to decrypt the 'done' message, the handshake has failed
    def send_done(self):
        self.send_aes(pad(b"done", 16))

    # request a payload then receive it
    def reqPayload(self):
        self.send_aes(b"bible.txt")
        res = self.get_aes()
        print(str(res, 'utf-8'))

    # wait for a request then send the requested file
    def awaitRequest(self):
        req = self.get_aes()
        fname = unpad(req.decode("utf-8"))
        f = open(fname, "rb")
        data = f.read()
        self.send_aes(data)

# pads byte arrays to a fixed length
def pad(line: bytes, padlen: int):
    elen = len(line) % padlen
    if elen: line += bytes(padlen - elen)
    return line

# remove padding from a string
def unpad(message):
    last_char = message[-1]
    if ord(last_char) < 32: return message.rstrip(last_char)
    else: return message

# wrapper for SHA256 hash
def hashBytes(data):
    digest = hashes.Hash(hashes.SHA256(), default_backend())
    digest.update(data)
    return digest.finalize()

client = Client()
client.handshake()
client.reqPayload()
client.awaitRequest()
