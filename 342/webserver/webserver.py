from socket import *
import sys # In order to terminate the program

serverSocket = socket(AF_INET, SOCK_STREAM)
# Prepare a server socket
serverSocket.bind(('', 4041))
serverSocket.listen(1)

while True:
    # Establish the connection
    print('Ready to serve...')
    connectionSocket, addr = serverSocket.accept()
    try:
        #Grab bytes
        message = connectionSocket.recv(1024)
        #Parse filename
        filename = message.split()[1]
        #Open file
        f = open(filename[1:])
        #Read file
        outputdata = f.read()
        #Send 200
        connectionSocket.send('\nHTTP/1.1 200 OK\n\n')
        #Send file Contents
        for i in range(0, len(outputdata)):
            connectionSocket.send(outputdata[i].encode())
        connectionSocket.send("\r\n".encode())
        #Close
        connectionSocket.close()
    except IOError:
        #Send 404 if file not found
        connectionSocket.send('\nHTTP/1.1 404 Not Found\n\n')
        connectionSocket.close()

#Close connection and exit
serverSocket.close()
sys.exit()
