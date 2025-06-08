from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation import TracerProvider
from utils.helper import get_phoenix_endpoint
from config import PROJECT_NAME


endpoint = get_phoenix_endpoint() + "v1/traces"
tracer_provider = register(project_name=PROJECT_NAME, endpoint=endpoint)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)
