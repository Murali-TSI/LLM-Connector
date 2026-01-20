# LLM Connector Examples

This folder contains examples demonstrating how to use the LLM Connector library.

## Setup

```bash
# Install the package with OpenAI support
uv sync --extra openai

# Set your API key
export OPENAI_API_KEY="your-api-key"
```

## Examples

| File | Description |
|------|-------------|
| `01_basic_chat.py` | Basic chat completion with string and structured messages |
| `02_streaming.py` | Real-time streaming responses |
| `03_tool_calling.py` | Function/tool calling with execution loop |
| `04_file_api.py` | File upload, download, list, and delete operations |
| `05_batch_api.py` | Batch processing for bulk requests (50% cost savings) |
| `06_multimodal.py` | Vision/image analysis with GPT-4o |
| `07_error_handling.py` | Comprehensive error handling patterns |

## Running Examples

```bash
# Run any example
uv run python examples/01_basic_chat.py

# Or with python directly
python examples/02_streaming.py
```

## Quick Reference

### Basic Chat
```python
from llm_connector import ConnectorFactory

connector = ConnectorFactory.create("openai")
response = connector.chat().invoke(messages="Hello!")
print(response.content)
```

### Streaming
```python
for chunk in connector.chat().invoke(messages="Tell me a story", stream=True):
    print(chunk.delta_content, end="", flush=True)
```

### Tool Calling
```python
tools = [{"name": "get_weather", "parameters": {...}}]
response = connector.chat().invoke(messages="What's the weather?", tools=tools)
if response.tool_calls:
    for tc in response.tool_calls:
        print(f"{tc.name}({tc.arguments})")
```

### File API
```python
file_api = connector.file()
file_id = file_api.upload(file=b"content", purpose="batch")
file_api.download(file_id=file_id)
file_api.delete(file_id=file_id)
```

### Batch API
```python
batch_api = connector.batch()
job = batch_api.create(file=jsonl_bytes, completion_window="24h")
status = batch_api.status(job.id)
results = batch_api.result(job.id)
```

### Error Handling
```python
from llm_connector.exceptions import RateLimitError, AuthenticationError

try:
    response = connector.chat().invoke(messages="Hello")
except RateLimitError as e:
    time.sleep(e.retry_after or 60)
except AuthenticationError:
    print("Check your API key")
```
