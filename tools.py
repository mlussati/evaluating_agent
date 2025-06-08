import pandas as pd
import duckdb
import json
from pydantic import BaseModel, Field
from opentelemetry.trace import StatusCode
from client import client
from tracing import tracer
from config import *

# SQL Generation
def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    formatted_prompt = SQL_GENERATION_PROMPT.format(prompt=prompt, columns=columns, table_name=table_name)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    return response.choices[0].message.content.strip().replace("```sql", "").replace("```", "")

@tracer.tool()
def lookup_sales_data(prompt: str) -> str:
    try:
        table_name = "sales"
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
        sql_query = generate_sql_query(prompt, df.columns, table_name)
        with tracer.start_as_current_span("execute_sql_query", openinference_span_kind="chain") as span:
            span.set_input(sql_query)
            result = duckdb.sql(sql_query).df()
            span.set_output(value=str(result))
            span.set_status(StatusCode.OK)
        return result.to_string()
    except Exception as e:
        return f"Error accessing data: {str(e)}"

@tracer.tool()
def analyze_sales_data(prompt: str, data: str) -> str:
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    return response.choices[0].message.content or "No analysis could be generated"

class VisualizationConfig(BaseModel):
    chart_type: str = Field(...)
    x_axis: str = Field(...)
    y_axis: str = Field(...)
    title: str = Field(...)

@tracer.chain()
def extract_chart_config(data: str, visualization_goal: str) -> dict:
    prompt = CHART_CONFIGURATION_PROMPT.format(data=data, visualization_goal=visualization_goal)
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=VisualizationConfig,
    )
    try:
        content = response.choices[0].message.content
        return {
            "chart_type": content.chart_type,
            "x_axis": content.x_axis,
            "y_axis": content.y_axis,
            "title": content.title,
            "data": data
        }
    except Exception:
        return {
            "chart_type": "line", 
            "x_axis": "date",
            "y_axis": "value",
            "title": visualization_goal,
            "data": data
        }

@tracer.chain()
def create_chart(config: dict) -> str:
    prompt = CREATE_CHART_PROMPT.format(config=config)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.replace("```python", "").replace("```", "").strip()

@tracer.tool()
def generate_visualization(data: str, visualization_goal: str) -> str:
    config = extract_chart_config(data, visualization_goal)
    return create_chart(config)

# tools list
TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_sales_data",
        "description": "Look up data from Store Sales dataset",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string", "description": "Prompt given by user."}}, "required": ["prompt"]}
    }},
    {"type": "function", "function": {
        "name": "analyze_sales_data",
        "description": "Analyze sales data to extract insights",
        "parameters": {"type": "object", "properties": {
            "data": {"type": "string"},
            "prompt": {"type": "string"}}, "required": ["data", "prompt"]}
    }},
    {"type": "function", "function": {
        "name": "generate_visualization",
        "description": "Generate Python code to create charts",
        "parameters": {"type": "object", "properties": {
            "data": {"type": "string"},
            "visualization_goal": {"type": "string"}}, "required": ["data", "visualization_goal"]}
    }}
]

IMPLEMENTATIONS = {
    "lookup_sales_data": lookup_sales_data,
    "analyze_sales_data": analyze_sales_data,
    "generate_visualization": generate_visualization
}