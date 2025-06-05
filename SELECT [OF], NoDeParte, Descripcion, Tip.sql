SELECT [OF], NoDeParte, Descripcion, Tipo, CASE WHEN  
       (SELECT TOP (1) MAX(Gag_Grupo) AS Maximo FROM MG_Calidad GROUP BY CodLinea, CodMaquina, TipoMaterial, [OF]  
       HAVING (CodLinea = N'10') AND (CodMaquina = 77) AND ([OF] = Especificaciones.[OF]) ORDER BY Maximo DESC) IS NULL THEN 0 ELSE  
       (SELECT TOP (1) MAX(Gag_Grupo) AS Maximo FROM MG_Calidad GROUP BY CodLinea, CodMaquina, TipoMaterial, [OF]  
       HAVING (CodLinea = N'10') AND (CodMaquina = 77) AND ([OF] = Especificaciones.[OF]) ORDER BY Maximo DESC) END AS Subgrupo,  
       (SELECT MAX(DISTINCT Fecha_Prueba) AS FechaUltimaPrueba FROM dbo.Cal_CTermico GROUP BY NoPte  
       HAVING (NoPte = dbo.Especificaciones.NoDeParte)) AS UltimaFechaCT, CodCliente, Espesor, X, Y, T_Muestra, RevDibujo, RevParte,  
       (SELECT NombreCorto FROM dbo.Clientes WHERE (CodCliente = dbo.Especificaciones.CodCliente)) AS ClienteNombreCorto, T_Muestra AS Expr1, NoDibujo  
       FROM dbo.Especificaciones GROUP BY NoDeParte, Descripcion, Tipo, Status, CodCliente, RevParte, Espesor, X, Y, T_Muestra, RevDibujo, NoDibujo, [OF]  
       HAVING  (Status = N'A') AND (Tipo = N'V') AND ([OF] = @laof) ORDER BY NoDeParte