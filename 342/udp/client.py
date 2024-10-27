from socket import *
from time import time, sleep

host = '128.180.120.92'
port = 4041
client = socket(AF_INET, SOCK_DGRAM)
client.settimeout(1)

for seq in range(1, 11):
    req = "udp_packet"
    try:
        start = time()
        client.sendto(req.encode(), (host, port))
        res, resAddr = client.recvfrom(1024)
        end = time()
        numBytes = len(res)
        res = res.decode()
        rtt = round((end-start)*10000, 1)
        print(str(numBytes) + " BYTES FROM " + str(resAddr) + ": " +
              "udp_seq=" + str(seq) + " time=" + str(rtt) + "ms message: " + res)
    except:
        print("UDP_SEQ=" + str(seq) + " timed out")
    sleep(.5)
client.close()
