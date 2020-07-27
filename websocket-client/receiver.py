#!/usr/bin/python 

import socket #import socket module

s = socket.socket() #create a socket object
host = 'localhost' #Host i.p
port = 12395  #Reserve a port for your service

s.connect((host,port))
print(s.recv(1024))
s.close
