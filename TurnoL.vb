Public Sub TurnoL()
    
    'Turno 1
    If (Time >= (CDate("07:00:00 a.m."))) And (Time <= (CDate("03:00:00 p.m."))) Then
        Turno = 1
        Fecha = Date
        HoraInicioTurno = CDate(Fecha & " 07:00:00 a.m.")
        HoraFinTurno = CDate(Fecha & " 03:00:00 p.m.")
    End If
    
     'Turno 2
    If (Time >= (CDate("03:00:00 p.m."))) And (Time <= (CDate("10:29:59 p.m."))) Then
        Turno = 2
        Fecha = Date
        HoraInicioTurno = CDate(Fecha & " 03:00:00 p.m.")
        HoraFinTurno = CDate(Fecha & " 10:29:59 p.m.")

    End If
     'Turno 3
    If (Time >= (CDate("10:30:00 p.m."))) And (Time <= (CDate("11:59:59 p.m."))) Then
        Turno = 3
        Fecha = Date
        HoraInicioTurno = CDate(Fecha & " 10:30:00 p.m.")
        HoraFinTurno = CDate((Fecha + 1) & " 06:59:59 a.m.")

    End If
    
    If (Time >= (CDate("00:00:00 a.m."))) And (Time <= (CDate("06:59:59 a.m."))) Then
        Turno = 3
        Fecha = Date - 1
        HoraInicioTurno = CDate(Fecha & " 10:30:00 p.m.")
        HoraFinTurno = CDate((Fecha + 1) & " 06:59:59 a.m.")

    End If
    
    If (Time > (CDate("12:00:00 a.m."))) And (Time <= (CDate("06:59:59 a.m."))) Then
        Turno = 3
        Fecha = Date - 1
        HoraInicioTurno = CDate(Fecha & " 10:30:00 p.m.")
        HoraFinTurno = CDate((Fecha + 1) & " 06:59:59 a.m.")
        
    End If
End Sub



