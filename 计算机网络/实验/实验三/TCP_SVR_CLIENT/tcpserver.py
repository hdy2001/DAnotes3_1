# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:46:09 2021

@author: hsc
"""

from socket import *
serverPort = 12000
serverSocket = socket(AF_INET,SOCK_STREAM)
serverSocket.bind(('127.0.0.1',serverPort))
serverSocket.listen(1)
print('The server is ready to receive')
while True:
     connectionSocket, addr = serverSocket.accept()
     sentence = connectionSocket.recv(1024).decode()
     capitalizedSentence = sentence.upper()
     print(capitalizedSentence)
     connectionSocket.send(capitalizedSentence.encode())
     connectionSocket.close()
