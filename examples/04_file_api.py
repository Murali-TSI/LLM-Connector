"""
File API Example

Demonstrates file upload, retrieval, listing, and deletion.
"""

import json
import tempfile
from pathlib import Path

from llm_connector import ConnectorFactory


def main():
    connector = ConnectorFactory.create("openai")
    file_api = connector.file()

    print("=" * 50)
    print("Example 1: Upload file from bytes")
    print("=" * 50)

    # Create sample JSONL content for batch processing
    batch_requests = [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 100,
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "What is the capital of Japan?"}
                ],
                "max_tokens": 100,
            },
        },
    ]

    # Convert to JSONL format
    jsonl_content = "\n".join(json.dumps(req) for req in batch_requests)

    # Upload from bytes
    file_id = file_api.upload(file=jsonl_content.encode("utf-8"), purpose="batch")
    print(f"Uploaded file ID: {file_id}")

    print()
    print("=" * 50)
    print("Example 2: Retrieve file metadata")
    print("=" * 50)

    file_info = file_api.retrieve(file_id=file_id)
    print(f"File ID: {file_info.id}")
    print(f"Filename: {file_info.filename}")
    print(f"Purpose: {file_info.purpose}")
    print(f"Size: {file_info.bytes} bytes")
    print(f"Created at: {file_info.created_at}")
    print(f"Status: {file_info.status}")

    print()
    print("=" * 50)
    print("Example 3: Upload file from path")
    print("=" * 50)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(jsonl_content)
        temp_path = f.name

    try:
        file_id_2 = file_api.upload(file=temp_path, purpose="batch")
        print(f"Uploaded from path: {file_id_2}")
    finally:
        Path(temp_path).unlink()  # Clean up temp file

    print()
    print("=" * 50)
    print("Example 4: List all files")
    print("=" * 50)

    files = file_api.list()
    print(f"Total files: {len(files)}")
    for f in files[:5]:  # Show first 5
        print(f"  - {f.id}: {f.filename} ({f.purpose})")

    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

    print()
    print("=" * 50)
    print("Example 5: List files by purpose")
    print("=" * 50)

    batch_files = file_api.list(purpose="batch")
    print(f"Batch files: {len(batch_files)}")
    for f in batch_files[:3]:
        print(f"  - {f.id}: {f.filename}")

    print()
    print("=" * 50)
    print("Example 6: Download file content")
    print("=" * 50)

    content = file_api.download(file_id=file_id)
    print(f"Downloaded {len(content)} bytes")
    print(f"Content preview: {content[:200].decode('utf-8')}...")

    print()
    print("=" * 50)
    print("Example 7: Delete files")
    print("=" * 50)

    # Clean up the files we created
    file_api.delete(file_id=file_id)
    print(f"Deleted: {file_id}")

    file_api.delete(file_id=file_id_2)
    print(f"Deleted: {file_id_2}")

    print()
    print("All examples completed!")


if __name__ == "__main__":
    main()
