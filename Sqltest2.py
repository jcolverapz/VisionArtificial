import pyodbc
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
#cursor.execute("INSERT INTO TblConteos('codlinea','TicketGem') VALUES (%s,%s)",('333','101010'))
#cursor.execute("INSERT INTO TblConteos('codlinea','TicketGem') VALUES (%s,%s)",('333','101010'))
        #sql = "INSERT INTO `TblConteos` (`codlinea`, `TicketGem`) VALUES (%s, %s)"
# sql = "INSERT INTO `TblConteos` (`codlinea`, `TicketGem`) VALUES (%s, %s)"
# cursor.execute(sql, ('333', '101010'))# sql = "INSERT INTO `` (`codlinea`, `TicketGem`) VALUES (%s, %s)"
# #        cursor.execute(sql, ('', ''))


SQLCommand = ("INSERT INTO TblConteos (codlinea,TicketGem,Nopuerto) VALUES (?,?,?);") 
sql = "UPDATE table SET telefonnummer = ? WHERE telefonnummer = ?"
Values = ['333','101010','0']

#Processing Query    
cursor.execute(SQLCommand,Values)

cnxn.commit()
print("Data Successfully Inserted")   
cnxn.close()