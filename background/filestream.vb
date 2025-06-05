FileStream se utiliza para leer y escribir archivos en un sistema de archivos, así como otros identificadores del sistema operativo relacionados con archivos tales como tuberías, entrada estándar, 
salida estándar. FileStream almacena en búfer la entrada y la salida para mejorar el rendimiento.


  Try       
            'Dim FS As New FileStream(path, fileMode)
            Dim fs As New FileStream(direccArchivo, FileMode.Open)
            fs.Position = 0
            Dim bytesDocumento(0 To fs.Length - 1) As Byte
            ' bytesDocumento(0 To fs.Length - 1)
            fs.Read(bytesDocumento, 0, fs.Length)

        Catch ex As Exception
            MsgBox(ex.Message)
            '  Exit Sub

        End Try

'FileStream.Lock(1, FileStream.Length)

'Dim fileLength As Integer = CInt(fs.Length)
'Dim buffer() As Byte = New Byte() {fileLength}