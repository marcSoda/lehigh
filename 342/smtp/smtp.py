from socket import *

msg = "\r\n You are getting a 100 in 342.".encode()
endMsg = "\r\n.\r\n".encode()

mailServer = "mail.cse.lehigh.edu"

clientSocket = socket(AF_INET, SOCK_STREAM)
clientSocket.connect((mailServer, 25))

recv = clientSocket.recv(1024).decode()
print(recv)
if recv[:3] != '220': print('220 reply not received from server.')

# Send HELO command and print server response.
heloCommand = 'HELO Alice\r\n'.encode()
clientSocket.send(heloCommand)
recv1 = clientSocket.recv(1024).decode()
print(recv1)
if recv1[:3] != '250': print('250 reply not received from server.')

# Send MAIL FROM command and print server response.
mailFrom = "MAIL FROM:<mol224@lehigh.edu> \r\n".encode()
clientSocket.send(mailFrom)
recv2 = clientSocket.recv(1024).decode()
print(recv2)
if recv2[:3] != '250': print('250 reply not received from server.')

# Send RCPT TO command and print server response.
rcptTo = "RCPT TO:<masa20@lehigh.edu> \r\n".encode()
clientSocket.send(rcptTo)
recv3 = clientSocket.recv(1024).decode()
print(recv3)
if recv3[:3] != '250': print('250 reply not received')

# Send DATA command and print server response.
data = "DATA \r\n".encode()
clientSocket.send(data)
recv4 = clientSocket.recv(1024).decode()
print(recv4)
if recv4[:3] != '354': print('250 reply not received from server.')

# Send message data.
clientSocket.send(msg)

# Message ends with a single period.
clientSocket.send(endMsg)

# Send QUIT command and get server response.
quit = "QUIT\r\n".encode()
clientSocket.send(quit)
recv5 = clientSocket.recv(1024).decode()
print(recv5)
clientSocket.close()
