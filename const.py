consulta_sql = """
SELECT c.Profissao,
       c.TempoProfissao,
       c.Renda,
       c.TipoResidencia,
       c.Escolaridade,
       c.Score,
       EXTRACT(YEAR FROM AGE(c.DataNascimento)) AS Idade,
       c.Dependentes,
       c.EstadoCivil,
       pf.NomeComercial AS Produto,
       pc.ValorSolicitado,
       pc.ValorTotalBem,
       CASE 
           WHEN COUNT(p.Status) FILTER (WHERE p.Status = 'Vencido') > 0 THEN 'ruim'
           ELSE 'bom'
       END AS Classe
FROM clientes c
JOIN PedidoCredito pc ON c.ClienteID = pc.ClienteID
JOIN ProdutosFinanciados pf ON pc.ProdutoID = pf.ProdutoID
LEFT JOIN ParcelasCredito p ON pc.SolicitacaoID = p.SolicitacaoID
WHERE pc.Status = 'Aprovado'
GROUP BY c.ClienteID, pf.NomeComercial, pc.ValorSolicitado, pc.ValorTotalBem
"""


scalers = [
    "tempoprofissao",
    "renda",
    "idade",
    "dependentes",
    "valorsolicitado",
    "valortotalbem",
    "proporcaosolicitadototal",
]

encoders = [
    "profissao",
    "tiporesidencia",
    "escolaridade",
    "score",
    "estadocivil",
    "produto",
]

profissoes_validas = [
    "Advogado",
    "Arquiteto",
    "Cientista de Dados",
    "Contador",
    "Dentista",
    "Empresário",
    "Engenheiro",
    "Médico",
    "Programador",
]
