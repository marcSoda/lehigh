import base64
import secrets
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

HEADERLEN = 8
HASHLEN = 32
SMALLCHOMPLEN = 8
BIGCHOMPLEN = 16384
HOST = ''
PORT = 4041

class Server:
    spri = None       # server private key
    spub = None       # server public key
    snonce = None     # server nonce
    ssock = None      # server socket
    cnonce = None     # client nonce
    csock = None      # client socket
    cipher = None     # AES encryption cypher

    def __init__(self):
        self.keygen()
        self.snonce = secrets.token_bytes(8)
        self.ssock = socket(AF_INET, SOCK_STREAM)
        self.ssock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.ssock.bind((HOST, PORT))
        self.ssock.listen()
        self.csock, _ = self.ssock.accept()

    # send based on a protocol that I created
    # first sends hash for clientside malformed data checking
    # then sends a header containing the number of bytes in payload
    # then sends the data
    def send(self, data):
        header = len(data).to_bytes(HEADERLEN, 'little')
        self.csock.send(header)
        self.csock.send(hashBytes(data))
        self.csock.send(data)

    # recv based on a protocol that I created
    # first receives a hash for local malformed data checking
    # then receives a header containing the number of bytes in payload
    # then receives the data in a buffered fashon
    def recv(self):
        data = bytes()
        header = self.csock.recv(HEADERLEN)
        hash = self.csock.recv(HASHLEN)
        datalen = int.from_bytes(header, 'little')
        chomplen = 0
        if (datalen > BIGCHOMPLEN): chomplen = BIGCHOMPLEN
        else: chomplen = SMALLCHOMPLEN
        while 1:
            d = self.csock.recv(chomplen)
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

    # generates a new public and private key for the server
    # normally this would be read from disk
    def keygen(self):
        self.spri = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        self.spub = self.spri.public_key()

    # perform tls handshake
    def handshake(self):
        print("beginning handshake")
        self.await_client_hello()
        print("received client hello")
        self.hello()
        print("sent server hello")
        pms = self.await_premaster()
        print("got premaster")
        self.constructMaster(pms)
        print("constructed master")
        self.send_done()
        print("sent done message")
        self.await_done()
        print("received done message")

    # waits for client hello
    # receives the client nonce. later used to generate the premaster secret on both sides
    # normally the client hello would contain TLS versions supported by the client as well as supported cipher suites
    def await_client_hello(self):
        self.cnonce = self.recv()

    # server hello
    # sends the server's pub key and nonce to the client.
    # pub key is used by the client to encrypt the premaster secret to be sent back to the server
    # nonce is used to calculate the premaster secret on both sides
    # normally this would contain an SSL cert that the client could verify, but my server does not have an SSL cert
    def hello(self):
        pem = self.spub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.send(self.snonce)
        self.send(pem)

    # receives the encrupted premaster secret from the client
    # can only be decrypted with server priv key
    # later used to construct the master secret
    def await_premaster(self):
        enc_pms = self.recv()
        return self.spri.decrypt(
            base64.b64decode(enc_pms),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    # using the premaster secret, client nonce, and server nonce, construct the master secret
    # this is the AES symmetric encryption key
    def constructMaster(self, pms):
        key = pms + self.cnonce + self.snonce
        iv = self.cnonce + self.snonce
        self.cipher = Cipher(algorithms.AES(key), modes.CBC(iv))

    # uses the generated symmetric encryption key, send the 'done' message to the client
    # this lets the client know that the server is ready for secure communication
    # if the client is unable to decrypt the 'done' message, the handshake has failed
    def send_done(self):
        self.send_aes(pad(b"done", 16))

    # await and decrypt the client 'done' message
    # signifies that the client is ready for secure communication
    # if the server is unable to decrypt the 'done' message, the handshake has failed
    def await_done(self):
        dec = self.get_aes()
        if (dec != pad(b"done", 16)):
            print("error in handshake")
            exit(1)

    # wait for a request then send the requested file
    def awaitRequest(self):
        req = self.get_aes()
        fname = unpad(req.decode("utf-8"))
        f = open(fname, "rb")
        data = f.read()
        self.send_aes(data)

    # request a payload then receive it
    def reqPayload(self):
        self.send_aes(b"clientfile.txt")
        res = self.get_aes()
        print(str(res, 'utf-8'))

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

server = Server()
server.handshake()
server.awaitRequest()
server.reqPayload()
