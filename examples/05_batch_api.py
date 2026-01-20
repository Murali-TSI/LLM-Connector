"""
Batch API Example

Demonstrates batch processing for bulk requests with 50% cost savings.
"""

import json
import time
from llm_connector import ConnectorFactory
from llm_connector.base import BatchStatus


def create_batch_requests(prompts: list[str], model: str = "gpt-4o-mini") -> str:
    """Create JSONL content for batch processing."""
    requests = []
    for i, prompt in enumerate(prompts):
        request = {
            "custom_id": f"request-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200
            }
        }
        requests.append(json.dumps(request))
    
    return "\n".join(requests)


def main():
    connector = ConnectorFactory.create("openai")
    batch_api = connector.batch()
    file_api = connector.file()

    print("=" * 50)
    print("Example 1: Create and submit a batch job")
    print("=" * 50)

    # Sample prompts to process
    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
    ]

    # Create batch file content
    jsonl_content = create_batch_requests(prompts)
    print(f"Created batch with {len(prompts)} requests")

    # Submit batch job
    batch_request = batch_api.create(
        file=jsonl_content.encode("utf-8"),
        completion_window="24h"
    )

    print(f"Batch job created!")
    print(f"  ID: {batch_request.id}")
    print(f"  Status: {batch_request.status.value}")
    print(f"  Input file: {batch_request.input_file_id}")
    print(f"  Endpoint: {batch_request.endpoint}")

    print()
    print("=" * 50)
    print("Example 2: Check batch status")
    print("=" * 50)

    job_id = batch_request.id
    
    # Poll for status (in real usage, you might use webhooks or longer intervals)
    max_checks = 30
    check_interval = 10  # seconds
    
    for i in range(max_checks):
        status = batch_api.status(job_id)
        print(f"[{i+1}/{max_checks}] Status: {status.status.value}", end="")
        
        if status.request_counts:
            counts = status.request_counts
            print(f" - Completed: {counts.get('completed', 0)}/{counts.get('total', 0)}", end="")
        
        print()
        
        # Check if job is done
        if status.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED]:
            break
        
        time.sleep(check_interval)
    
    print(f"\nFinal status: {status.status.value}")

    print()
    print("=" * 50)
    print("Example 3: Retrieve batch results")
    print("=" * 50)

    if status.status == BatchStatus.COMPLETED:
        result = batch_api.result(job_id)
        
        print(f"Output file: {result.output_file_id}")
        print(f"Total records: {len(result.records)}")
        print()
        
        for record in result.records:
            custom_id = record.get("custom_id", "unknown")
            response = record.get("response", {})
            body = response.get("body", {})
            
            if "choices" in body:
                content = body["choices"][0]["message"]["content"]
                print(f"{custom_id}: {content[:100]}...")
            elif "error" in record:
                print(f"{custom_id}: ERROR - {record['error']}")
    else:
        print(f"Batch did not complete successfully. Status: {status.status.value}")
        
        # Check for error file
        if status.error_file_id:
            error_content = file_api.download(file_id=status.error_file_id)
            print(f"Error details: {error_content.decode('utf-8')[:500]}")

    print()
    print("=" * 50)
    print("Example 4: List recent batch jobs")
    print("=" * 50)

    batches = batch_api.list(limit=5)
    print(f"Recent batch jobs ({len(batches)}):")
    
    for batch in batches:
        print(f"  - {batch.id}")
        print(f"    Status: {batch.status.value}")
        print(f"    Created: {batch.timestamps.created_at}")
        if batch.timestamps.completed_at:
            print(f"    Completed: {batch.timestamps.completed_at}")
        print()

    print()
    print("=" * 50)
    print("Example 5: Cancel a batch job (demo)")
    print("=" * 50)

    # Create another batch to demonstrate cancellation
    small_batch = create_batch_requests(["Test prompt 1", "Test prompt 2"])
    new_batch = batch_api.create(
        file=small_batch.encode("utf-8"),
        completion_window="24h"
    )
    
    print(f"Created batch: {new_batch.id}")
    
    # Cancel it immediately
    if new_batch.status in [BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS]:
        cancelled = batch_api.cancel(new_batch.id)
        print(f"Cancelled batch: {cancelled.id}")
        print(f"Status after cancel: {cancelled.status.value}")
    else:
        print(f"Batch already in terminal state: {new_batch.status.value}")

    print()
    print("=" * 50)
    print("Batch processing examples completed!")
    print("=" * 50)
    print()
    print("Tips:")
    print("- Batch API offers 50% cost savings compared to real-time API")
    print("- Results are available within 24 hours")
    print("- Use for non-time-sensitive bulk processing")
    print("- Each batch can contain up to 50,000 requests")


if __name__ == "__main__":
    main()
