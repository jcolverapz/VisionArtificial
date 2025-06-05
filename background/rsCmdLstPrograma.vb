rsCmdLstPrograma


CNN.CmdLstPrograma (Fecha - 2), (Fecha), (CodLinea)

'CNN.CmdLstPrograma (Fecha - 5), (Fecha + 1), (CodLinea)

'CNN.CmdLstPrograma (Fecha - 3), (Fecha + 3), (CodLinea)

If CNN.rsCmdLstPrograma.EOF <> True Then
    Me.LstReqAT.Enabled = True
    Do While CNN.rsCmdLstPrograma.EOF <> True
        
        Resp = CNN.rsCmdLstPrograma.RecordCount
        
        Me.LblSinJC.Visible = False
        Me.LstReqAT.Visible = True
        
        Me.CmdArriba.Visible = True
        Me.CmdAbajo.Visible = True
        
 CNN.CmdBuscaNoPartexOF (CNN.rsCmdLstPrograma!OF)
                If CNN.rsCmdBuscaNoPartexOF.EOF <> True Then
                    Me.LstReqAT.ListItems.Item(i).SubItems(4) = CNN.rsCmdBuscaNoPartexOF!nodeparte
                Else
                    Me.LstReqAT.ListItems.Item(i).SubItems(4) = CNN.rsCmdLstPrograma!OF
                End If
                CNN.rsCmdBuscaNoPartexOF.Close
                Me.LstReqAT.ListItems.Item(i).SubItems(5) = "X"  'CNN.rsCmdLstPrograma!tpiezas


                of=7978


                  SQL = " SELECT COUNT(Tbl_Enc_EntAcum.IdAcum) AS T, Tbl_Enc_EntAcum.IdJC,"
                SQL = SQL & "     Tbl_Enc_EntAcum.Turno"
                SQL = SQL & " FROM Tbl_Enc_EntAcum INNER JOIN"
                SQL = SQL & "     Tbl_Det_EntAcum ON"
                SQL = SQL & "     Tbl_Enc_EntAcum.IdAcum = Tbl_Det_EntAcum.IdAcum"
                SQL = SQL & " WHERE (Tbl_Enc_EntAcum.Enc_Status = N'AC') AND"
                SQL = SQL & "     (Tbl_Enc_EntAcum.IdJC = " & CNN.rsCmdLstPrograma!OMA_ID & ") AND (Tbl_Enc_EntAcum.Turno = " & CNN.rsCmdLstPrograma!Turno & ")"
                SQL = SQL & " GROUP BY Tbl_Enc_EntAcum.IdJC, Tbl_Enc_EntAcum.Turno"


                  
                CNN.rsCmdTFoliosxJC.Open SQL
                If CNN.rsCmdTFoliosxJC.EOF <> True Then
                    Me.LstReqAT.ListItems.Item(i).SubItems(6) = CNN.rsCmdTFoliosxJC!t
                Else
                    Me.LstReqAT.ListItems.Item(i).SubItems(6) = ""
                End If
                CNN.rsCmdTFoliosxJC.Close
                

                
                
                Me.LstReqAT.ListItems.Item(i).SubItems(7) = Mid(CNN.rsCmdLstPrograma!CodVidrio, 1, 3)