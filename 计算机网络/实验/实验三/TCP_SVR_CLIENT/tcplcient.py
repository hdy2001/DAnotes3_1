# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:44:07 2021

@author: hsc
"""

from socket import *
import time
serverName = '127.0.0.1'
serverPort = 12000
i=0
while True:
    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverName,serverPort))
    sentence = input('Input lowercase sentence:')
    clientSocket.send(sentence.encode())
    modifiedSentence = clientSocket.recv(1024)
    print ('From Server:', modifiedSentence.decode())
    time.sleep(1)
clientSocket.close()
