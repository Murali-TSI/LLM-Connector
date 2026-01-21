"""
Async Tool Calling Example

Demonstrates async function/tool calling with the OpenAI provider.
"""

import json
import asyncio
from llm_connector import ConnectorFactory, TextBlock, Role
from llm_connector import AssistantMessage, ToolMessage, UserMessage, ToolCall


def get_weather(location: str, unit: str = "celsius") -> dict:
    """Simulated weather API call."""
    weather_data = {
        "Tokyo": {"temp": 22, "condition": "Sunny"},
        "London": {"temp": 15, "condition": "Cloudy"},
        "New York": {"temp": 18, "condition": "Partly cloudy"},
    }
    
    data = weather_data.get(location, {"temp": 20, "condition": "Unknown"})
    
    if unit == "fahrenheit":
        data["temp"] = (data["temp"] * 9/5) + 32
    
    return {
        "location": location,
        "temperature": data["temp"],
        "unit": unit,
        "condition": data["condition"],
    }


def get_time(timezone: str) -> dict:
    """Simulated time API call."""
    from datetime import datetime, timedelta
    
    offsets = {
        "UTC": 0,
        "EST": -5,
        "PST": -8,
        "JST": 9,
        "GMT": 0,
    }
    
    offset = offsets.get(timezone.upper(), 0)
    current_time = datetime.utcnow() + timedelta(hours=offset)
    
    return {
        "timezone": timezone,
        "time": current_time.strftime("%H:%M:%S"),
        "date": current_time.strftime("%Y-%m-%d"),
    }


# Tool definitions
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g., Tokyo, London, New York"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_time",
        "description": "Get the current time in a given timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone, e.g., UTC, EST, PST, JST"
                }
            },
            "required": ["timezone"]
        }
    }
]

available_functions = {
    "get_weather": get_weather,
    "get_time": get_time,
}


async def main():
    connector = ConnectorFactory.create("openai")
    chat = connector.async_chat()

    print("=" * 50)
    print("Example 1: Async single tool call")
    print("=" * 50)

    response = await chat.invoke(
        messages="What's the weather like in Tokyo?",
        tools=tools,
    )

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool called: {tool_call.name}")
            print(f"Arguments: {tool_call.arguments}")
            
            func = available_functions[tool_call.name]
            result = func(**tool_call.arguments)
            print(f"Result: {result}")
    else:
        print(f"Response: {response.content}")

    print()

    # Full async conversation with tool execution
    print("=" * 50)
    print("Example 2: Async full conversation with tool execution")
    print("=" * 50)

    messages = [
        UserMessage(
            role=Role.USER,
            content=[TextBlock(text="What's the weather in London and what time is it in JST?")]
        )
    ]

    response = await chat.invoke(messages=messages, tools=tools)

    if response.tool_calls:
        print(f"Model wants to call {len(response.tool_calls)} tool(s)")
        
        messages.append(
            AssistantMessage(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id=tc.id,
                        name=tc.name,
                        arguments=tc.arguments
                    )
                    for tc in response.tool_calls
                ]
            )
        )

        for tool_call in response.tool_calls:
            print(f"  Executing: {tool_call.name}({tool_call.arguments})")
            
            func = available_functions[tool_call.name]
            result = func(**tool_call.arguments)
            
            messages.append(
                ToolMessage(
                    role=Role.TOOL,
                    tool_call_id=tool_call.id,
                    content=[TextBlock(text=json.dumps(result))]
                )
            )

        final_response = await chat.invoke(messages=messages, tools=tools)
        print(f"\nFinal response: {final_response.content}")
    else:
        print(f"Response: {response.content}")

    print()

    # Async streaming with tools
    print("=" * 50)
    print("Example 3: Async streaming with tool calls")
    print("=" * 50)

    stream = await chat.invoke(
        messages="Tell me the weather in New York",
        tools=tools,
        stream=True,
    )

    collected_tool_calls = {}
    
    async for chunk in stream:
        if chunk.delta_content:
            print(chunk.delta_content, end="", flush=True)
        
        if chunk.delta_tool_calls:
            for tc_delta in chunk.delta_tool_calls:
                idx = tc_delta.index
                
                if idx not in collected_tool_calls:
                    collected_tool_calls[idx] = {
                        "id": "",
                        "name": "",
                        "arguments": ""
                    }
                
                if tc_delta.id:
                    collected_tool_calls[idx]["id"] = tc_delta.id
                if tc_delta.name:
                    collected_tool_calls[idx]["name"] = tc_delta.name
                if tc_delta.arguments:
                    collected_tool_calls[idx]["arguments"] += tc_delta.arguments

    if collected_tool_calls:
        print("\nTool calls from stream:")
        for idx, tc in collected_tool_calls.items():
            print(f"  [{idx}] {tc['name']}: {tc['arguments']}")
            
            func = available_functions[tc["name"]]
            args = json.loads(tc["arguments"])
            result = func(**args)
            print(f"       Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
