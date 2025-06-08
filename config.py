MODEL = "gpt-4o-mini"
PROJECT_NAME = "tracing-agent"
TRANSACTION_DATA_FILE_PATH = 'data/Store_Sales_Price_Elasticity_Promotions_Data.parquet'
SQL_GENERATION_PROMPT = """
Gere uma consulta SQL com base no prompt. Não responda com nada além da consulta SQL.
O prompt é: {prompt}

As colunas disponíveis são: {columns}
O nome da tabela é: {table_name}
"""
DATA_ANALYSIS_PROMPT = """
Analise os seguintes dados: {data}
Seu trabalho é responder à seguinte pergunta: {prompt}
"""
CHART_CONFIGURATION_PROMPT = """
Gere uma configuração de gráfico com base nesses dados: {data}
O objetivo é mostrar: {visualization_goal}
"""
CREATE_CHART_PROMPT = """
Escreva código Python para criar um gráfico com base na seguinte configuração.
Retorne apenas o código, sem nenhum outro texto.
configuração: {config}
"""
SYSTEM_PROMPT = """
Você é um assistente útil que pode responder perguntas sobre o conjunto de dados de Promoções de Elasticidade de Preço de Vendas em Loja.
"""
