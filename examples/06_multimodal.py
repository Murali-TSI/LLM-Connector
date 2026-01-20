"""
Multimodal (Vision) Example

Demonstrates sending images to the model for analysis.
"""

import base64
from pathlib import Path

from llm_connector import ConnectorFactory
from llm_connector import UserMessage, TextBlock, ImageBlock, Role


def encode_image_to_data_url(image_path: str) -> str:
    """Convert a local image to a data URL."""
    path = Path(image_path)
    
    # Determine MIME type
    suffix = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/jpeg")
    
    # Read and encode
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    
    return f"data:{mime_type};base64,{encoded}"


def main():
    connector = ConnectorFactory.create("openai")
    chat = connector.chat()

    print("=" * 50)
    print("Example 1: Analyze image from URL")
    print("=" * 50)

    # Using a public image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

    message = UserMessage(
        role=Role.USER,
        content=[
            TextBlock(text="What animal is in this image? Describe it briefly."),
            ImageBlock(url=image_url, detail="low"),
        ]
    )

    response = chat.invoke(
        messages=message,
        model="gpt-4o-mini",  # Vision-capable model
        max_tokens=300,
    )

    print(f"Response: {response.content}")
    print(f"Tokens used: {response.usage.total_tokens}")

    print()
    print("=" * 50)
    print("Example 2: Image with high detail")
    print("=" * 50)

    message = UserMessage(
        role=Role.USER,
        content=[
            TextBlock(text="Describe this image in detail, including colors, composition, and any text visible."),
            ImageBlock(url=image_url, detail="high"),
        ]
    )

    response = chat.invoke(
        messages=message,
        model="gpt-4o-mini",
        max_tokens=500,
    )

    print(f"Response: {response.content}")

    print()
    print("=" * 50)
    print("Example 3: Multiple images comparison")
    print("=" * 50)

    image_url_1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    image_url_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"

    message = UserMessage(
        role=Role.USER,
        content=[
            TextBlock(text="Compare these two images. What animals are shown and how do they differ?"),
            ImageBlock(url=image_url_1, detail="low"),
            ImageBlock(url=image_url_2, detail="low"),
        ]
    )

    response = chat.invoke(
        messages=message,
        model="gpt-4o-mini",
        max_tokens=400,
    )

    print(f"Response: {response.content}")

    print()
    print("=" * 50)
    print("Example 4: Streaming with image input")
    print("=" * 50)

    message = UserMessage(
        role=Role.USER,
        content=[
            TextBlock(text="Write a short poem inspired by this image."),
            ImageBlock(url=image_url, detail="low"),
        ]
    )

    stream = chat.invoke(
        messages=message,
        model="gpt-4o-mini",
        max_tokens=200,
        stream=True,
    )

    for chunk in stream:
        if chunk.delta_content:
            print(chunk.delta_content, end="", flush=True)
    
    print()

    print()
    print("=" * 50)
    print("Example 5: Local image (base64 encoded)")
    print("=" * 50)
    print("Note: This example shows the pattern for local images.")
    print("Uncomment and provide a valid image path to test.")
    print()
    
    # Uncomment below to test with a local image:
    # local_image_path = "/path/to/your/image.jpg"
    # data_url = encode_image_to_data_url(local_image_path)
    # 
    # message = UserMessage(
    #     role=Role.USER,
    #     content=[
    #         TextBlock(text="What do you see in this image?"),
    #         ImageBlock(url=data_url, detail="auto"),
    #     ]
    # )
    # 
    # response = chat.invoke(messages=message, model="gpt-4o-mini")
    # print(f"Response: {response.content}")

    print("Code pattern for local images:")
    print("""
    from pathlib import Path
    import base64
    
    # Read and encode image
    with open("image.jpg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    
    data_url = f"data:image/jpeg;base64,{encoded}"
    
    message = UserMessage(
        role=Role.USER,
        content=[
            TextBlock(text="Describe this image"),
            ImageBlock(url=data_url, detail="auto"),
        ]
    )
    """)


if __name__ == "__main__":
    main()
