import pymysql

conn = pymysql.connect(
    host='localhost',
    user='root',
    password='password',
    db='mydatabase',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)
    # _cadenaConexion As String = "Data Source=10.18.172.2;Initial Catalog=Gemtron;Persist Security Info=True;;"


try:
    with conn.cursor() as cursor:
        # Create a new record
        #sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
        sql = "INSERT INTO `TblConteos` (`codlinea`, `TicketGem`) VALUES (%s, %s)"
        cursor.execute(sql, ('333', '101010'))

    # Commit changes
    conn.commit()

    print("Record inserted successfully")
finally:
    conn.close()
    
    
    
        # _comando.CommandText = "INSERT INTO TblConteos  ([codlinea], [NoPuerto], [Timestamp], [pz], [TicketGem], [JobCard]) VALUES (@His_Id, @Rep_Titulo, @Rep_No, @Rep_Status)"

        # _comando.Connection.Open()

        # ' _comando.CommandType = CommandType.StoredProcedure

        # _comando.Parameters.Clear()
        # _comando.CommandType = CommandType.Text

        # _comando.Parameters.AddWithValue("@Rep_Titulo", Reporte.descripcion)
        # _comando.Parameters.AddWithValue("@His_Id", Reporte.His_Id)
        # _comando.Parameters.AddWithValue("@Rep_No", Reporte.No)
        # _comando.Parameters.AddWithValue("@Rep_Status", Reporte.Status)

# SELECT        TOP (200) Id, , , , , , 
# FROM            TblConteos