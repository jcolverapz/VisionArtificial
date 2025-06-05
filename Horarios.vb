
Public Sub Horarios()
'On Error GoTo ErrHandler
'If Conexion = False Then Exit Sub

CNN.rsCmdTblHorarios.Open
Do While CNN.rsCmdTblHorarios.EOF <> True
        FechaHorario_Ini = CDate("01/01/1900 " & Mid(CNN.rsCmdTblHorarios!HoraInicio, 1, 8))
        FechaHorarioSingle_Ini = CSng(CDate("01/01/1900 " & Mid(CNN.rsCmdTblHorarios!HoraInicio, 1, 8)))
        FechaHorario_Fin = CDate("01/01/1900 " & Mid(CNN.rsCmdTblHorarios!HoraTermino, 1, 8))
        FechaHorarioSingle_Fin = CSng(CDate("01/01/1900 " & Mid(CNN.rsCmdTblHorarios!HoraTermino, 1, 8)))
        FechaHorario = CDate("01/01/1900 " & Mid(Time, 1, 8) & " PM")
        FechaHorarioSingle = CSng(CDate("01/01/1900 " & Time & ""))
        If (FechaHorarioSingle_Ini <= FechaHorarioSingle) And (FechaHorarioSingle_Fin >= FechaHorarioSingle) Then
              IdHorario = CNN.rsCmdTblHorarios!IdHorario
              CNN.rsCmdTblHorarios.MoveLast
        Else
             If CNN.rsCmdTblHorarios!IdHorario = 23 Then
                   ''' para de las  22:30 a las 00:00
                   If FechaHorarioSingle_Ini <= FechaHorarioSingle Then
                        If FechaHorarioSingle < 3 Then
                              IdHorario = CNN.rsCmdTblHorarios!IdHorario
                              CNN.rsCmdTblHorarios.MoveLast
                        End If
                   End If
                  ''' para de las  00:00 a las 00:15
                   If FechaHorarioSingle_Fin >= FechaHorarioSingle Then
                             IdHorario = CNN.rsCmdTblHorarios!IdHorario
                             CNN.rsCmdTblHorarios.MoveLast
                   End If
             End If
        End If
   CNN.rsCmdTblHorarios.MoveNext
Loop
CNN.rsCmdTblHorarios.Close

'ErrHandler:
'If Err.Number = -2147467259 Then
' Me.lblactualiza.Caption = "Err: " & Format(Now, "dd/MM hh:mm")
  ' Exit Sub
    
'End If

End Sub


