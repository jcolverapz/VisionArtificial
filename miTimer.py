# Import the time library
import time
import pyodbc

def Guardar(seconds):
	user='MXAPDAPP'
	password='Sch0tt!'
	database='Gemtron'
	port='1433'
	TDS_Version='8.0'
	server='10.18.172.2'
	driver='SQL SERVER'
	con_string='UID=%s;PWD=%s;DATABASE=%s;PORT=%s;TDS=%s;SERVER=%s;driver=%s' % (user,password, database,port,TDS_Version,server,driver)
	cnxn=pyodbc.connect(con_string)
	cursor=cnxn.cursor()

	cursor.execute("""UPDATE Tbl_Conteos_Estados set PzMov=? where Con_Linea=331""", (seconds))

	cnxn.commit()
	cnxn.close()

#from Guardar import *
def timer1():
    # Calculate the start time
    start = time.time()

    running = True
    seconds = 0
    end = 60

    while (running):
        time.sleep(1)
        seconds +=1
        if seconds >= end:
            running = False
            #GuardarSQL.update(seconds)
            Guardar(seconds)
            
timer1()