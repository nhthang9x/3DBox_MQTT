#!/usr/bin/python 

import socket #import socket module

s = socket.socket() #create a socket object
host = '192.168.100.38' #Host i.p
port = 3030  #Reserve a port for your service

s.connect((host,port))


s.close
