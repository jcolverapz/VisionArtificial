#import dbconfigs
import pyodbc
import pandas as pd

cn1= pyodbc.connect('r'DRIVER=SQL SERVER;'
                    r'Database=Gemtron;'
                    f'User ID={db};'
                    f'Password={db}')
                    
                    
for row in df.itertuples():
    qry = f"""INSERT INTO ClientData(ClientName, ClientSurname,ClientTel)
     
    
    csr.execute(qry)
    
    csr.commit()
    csr.close()