import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
downloads_dir = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd().parent
sys.path.insert(0, str(downloads_dir))

from llm_connector import ConnectorFactory

connector = ConnectorFactory.create("openai", config={"api_key": os.getenv("OPENAI_API_KEY")})

response = connector.chat().invoke(messages="Hello, how are you?")
print(response.content)

for chunk in connector.chat().invoke(messages="Tell me a story", stream=True):
    print(chunk.delta_content, end="", flush=True)

batch = connector.batch().create([
    {"type": "chat", "messages": "What's the weather like today?"},
    {"type": "chat", "messages": "Give me a recipe for pancakes."}
])