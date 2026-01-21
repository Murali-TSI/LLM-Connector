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

### Synchronous Examples

| File | Description |
|------|-------------|
| `01_basic_chat.py` | Basic chat completion with string and structured messages |
| `02_streaming.py` | Real-time streaming responses |
| `03_tool_calling.py` | Function/tool calling with execution loop |
| `04_file_api.py` | File upload, download, list, and delete operations |
| `05_batch_api.py` | Batch processing for bulk requests (50% cost savings) |
| `06_multimodal.py` | Vision/image analysis with GPT-4o |
| `07_error_handling.py` | Comprehensive error handling patterns |

### Asynchronous Examples

| File | Description |
|------|-------------|
| `08_async_basic_chat.py` | Async chat completion with await |
| `09_async_streaming.py` | Async streaming with async for loops |
| `10_async_tool_calling.py` | Async function/tool calling |
| `11_async_concurrent.py` | Concurrent requests for better performance |

## Running Examples

```bash
# Run any example
uv run python examples/01_basic_chat.py

# Run async examples
uv run python examples/08_async_basic_chat.py
```

## Quick Reference

### Sync Chat
```python
from llm_connector import ConnectorFactory

connector = ConnectorFactory.create("openai")
response = connector.chat().invoke(messages="Hello!")
print(response.content)
```

### Async Chat
```python
import asyncio
from llm_connector import ConnectorFactory

async def main():
    connector = ConnectorFactory.create("openai")
    response = await connector.async_chat().invoke(messages="Hello!")
    print(response.content)

asyncio.run(main())
```

### Sync Streaming
```python
for chunk in connector.chat().invoke(messages="Tell me a story", stream=True):
    print(chunk.delta_content, end="", flush=True)
```

### Async Streaming
```python
stream = await connector.async_chat().invoke(messages="Tell me a story", stream=True)
async for chunk in stream:
    print(chunk.delta_content, end="", flush=True)
```

### Concurrent Async Requests
```python
import asyncio

async def main():
    connector = ConnectorFactory.create("openai")
    chat = connector.async_chat()
    
    questions = ["Q1?", "Q2?", "Q3?"]
    tasks = [chat.invoke(messages=q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    for r in results:
        print(r.content)

asyncio.run(main())
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

### Async File API
```python
file_api = connector.async_file()
file_id = await file_api.upload(file=b"content", purpose="batch")
content = await file_api.download(file_id=file_id)
await file_api.delete(file_id=file_id)
```

### Batch API
```python
batch_api = connector.batch()
job = batch_api.create(file=jsonl_bytes, completion_window="24h")
status = batch_api.status(job.id)
results = batch_api.result(job.id)
```

### Async Batch API
```python
batch_api = connector.async_batch()
job = await batch_api.create(file=jsonl_bytes, completion_window="24h")
status = await batch_api.status(job.id)
results = await batch_api.result(job.id)
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

## API Reference

### Connector Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `chat()` | `ChatCompletion` | Sync chat interface |
| `batch()` | `BatchProcess` | Sync batch interface |
| `file()` | `FileAPI` | Sync file interface |
| `async_chat()` | `AsyncChatCompletion` | Async chat interface |
| `async_batch()` | `AsyncBatchProcess` | Async batch interface |
| `async_file()` | `AsyncFileAPI` | Async file interface |
