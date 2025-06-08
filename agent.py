from tracing import tracer
from client import client
from tools import TOOLS, IMPLEMENTATIONS
from config import SYSTEM_PROMPT
from opentelemetry.trace import StatusCode
import json

@tracer.chain()
def handle_tool_calls(tool_calls, messages):
    for call in tool_calls:
        func = IMPLEMENTATIONS[call.function.name]
        args = json.loads(call.function.arguments)
        result = func(**args)
        messages.append({"role": "tool", "content": result, "tool_call_id": call.id})
    return messages

def run_agent(messages):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not any(m.get("role") == "system" for m in messages if isinstance(m, dict)):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    while True:
        with tracer.start_as_current_span("router_call", openinference_span_kind="chain") as span:
            span.set_input(value=messages)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message.model_dump()
            messages.append(msg)
            calls = response.choices[0].message.tool_calls
            if calls:
                messages = handle_tool_calls(calls, messages)
                span.set_output(value=calls)
            else:
                span.set_output(value=response.choices[0].message.content)
                return response.choices[0].message.content

def start_main_span(messages):
    with tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as span:
        span.set_input(value=messages)
        result = run_agent(messages)
        span.set_output(value=result)
        span.set_status(StatusCode.OK)
        return result
