import cv2
import numpy as np 
import imutils 
import os


# Datos = 'objects'
# if not os.path.exists(Datos):
#     print('Carpeta creada: ',Datos)
#     os.makedirs(Datos)
# counter=0
# i=0
Datos = '\\10.18.172.30\Departments\IT\Read\Reportes_Tickets\Nuevo'

#if os.path.isdir == ('\\10.18.172.30\Departments\IT\Read\Reportes_Tickets\Nuevo'):
if os.path.isdir == (Datos):
	print(f'Si existe')

else:
    print(f'no existe, se crea')
    #os.mkdir('\\10.18.172.30\Departments\IT\Read\Reportes_Tickets\Nuevo')
    #os.makedirs(Datos)
    os.makedirs('\\10.18.172.30\Departments\IT\\Read\Reportes_Tickets\Nuevo')
    
		#os.mkdir('nombre_carpeta\dia'')}