"""
Error Handling Example

Demonstrates proper error handling for various failure scenarios.
"""

import os
from llm_connector import ConnectorFactory
from llm_connector.exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ContextLengthExceededError,
    ContentFilterError,
    APIError,
    ProviderNotSupportedError,
    ProviderImportError,
    BatchError,
    FileError,
)


def example_authentication_error():
    """Handle invalid API key."""
    print("=" * 50)
    print("Example 1: Authentication Error")
    print("=" * 50)
    
    try:
        # Temporarily use invalid key
        connector = ConnectorFactory.create("openai", config={"api_key": "invalid-key"})
        connector.chat().invoke(messages="Hello")
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        print("Solution: Check your OPENAI_API_KEY environment variable")
    print()


def example_rate_limit_error():
    """Handle rate limiting."""
    print("=" * 50)
    print("Example 2: Rate Limit Error (simulated)")
    print("=" * 50)
    
    # This would happen with excessive requests
    # Simulating the error handling pattern:
    try:
        # In real scenario, this might happen after many rapid requests
        raise RateLimitError("Rate limit exceeded", retry_after=30.0)
    except RateLimitError as e:
        print(f"Rate limited: {e}")
        if e.retry_after:
            print(f"Retry after: {e.retry_after} seconds")
            # time.sleep(e.retry_after)  # Would wait here in real code
    print()


def example_context_length_error():
    """Handle context length exceeded."""
    print("=" * 50)
    print("Example 3: Context Length Exceeded")
    print("=" * 50)
    
    connector = ConnectorFactory.create("openai")
    
    try:
        # Create a very long message that exceeds context
        long_message = "Hello " * 100000  # ~500k tokens
        connector.chat().invoke(messages=long_message)
    except ContextLengthExceededError as e:
        print(f"Context too long: {e}")
        print("Solution: Reduce input size or use a model with larger context")
    except InvalidRequestError as e:
        # Sometimes this is caught as InvalidRequestError
        print(f"Invalid request (likely context length): {e}")
    print()


def example_invalid_request():
    """Handle invalid request parameters."""
    print("=" * 50)
    print("Example 4: Invalid Request")
    print("=" * 50)
    
    connector = ConnectorFactory.create("openai")
    
    try:
        # Invalid model name
        connector.chat().invoke(
            messages="Hello",
            model="nonexistent-model-xyz"
        )
    except InvalidRequestError as e:
        print(f"Invalid request: {e}")
    except APIError as e:
        print(f"API error (invalid model): {e}")
        print(f"Status code: {e.status_code}")
    print()


def example_provider_not_supported():
    """Handle unsupported provider."""
    print("=" * 50)
    print("Example 5: Provider Not Supported")
    print("=" * 50)
    
    try:
        ConnectorFactory.create("unsupported_provider")
    except ProviderNotSupportedError as e:
        print(f"Provider error: {e}")
        print(f"Available providers: {ConnectorFactory.supported_providers()}")
    print()


def example_comprehensive_error_handling():
    """Comprehensive error handling pattern."""
    print("=" * 50)
    print("Example 6: Comprehensive Error Handling Pattern")
    print("=" * 50)
    
    def safe_chat(messages: str, max_retries: int = 3):
        """Safely make a chat request with retries."""
        import time
        
        connector = ConnectorFactory.create("openai")
        
        for attempt in range(max_retries):
            try:
                response = connector.chat().invoke(messages=messages)
                return response.content
                
            except AuthenticationError:
                print("❌ Invalid API key - cannot retry")
                raise
                
            except RateLimitError as e:
                wait_time = e.retry_after or (2 ** attempt)  # Exponential backoff
                print(f"⏳ Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise
                    
            except ContextLengthExceededError:
                print("❌ Message too long - cannot retry")
                raise
                
            except ContentFilterError:
                print("❌ Content blocked by safety filter")
                raise
                
            except APIError as e:
                print(f"⚠️ API error (status {e.status_code}), attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
                    
            except Exception as e:
                print(f"❌ Unexpected error: {type(e).__name__}: {e}")
                raise
        
        return None

    # Test the pattern
    try:
        result = safe_chat("What is 2+2?", max_retries=3)
        if result:
            print(f"✅ Success: {result}")
    except Exception as e:
        print(f"Failed after retries: {e}")
    print()


def example_batch_and_file_errors():
    """Handle batch and file API errors."""
    print("=" * 50)
    print("Example 7: Batch and File Error Handling")
    print("=" * 50)
    
    connector = ConnectorFactory.create("openai")
    
    # File error
    try:
        connector.file().retrieve(file_id="nonexistent-file-id")
    except FileError as e:
        print(f"File error: {e}")
    
    # Batch error
    try:
        connector.batch().status("nonexistent-batch-id")
    except BatchError as e:
        print(f"Batch error: {e}")
    except APIError as e:
        print(f"API error: {e}")
    print()


def main():
    print("LLM Connector Error Handling Examples")
    print("=" * 50)
    print()
    
    # Skip auth error example if no key (would fail immediately)
    if os.environ.get("OPENAI_API_KEY"):
        # example_authentication_error()  # Uncomment to test with invalid key
        pass
    
    example_rate_limit_error()
    
    if os.environ.get("OPENAI_API_KEY"):
        example_context_length_error()
        example_invalid_request()
    
    example_provider_not_supported()
    
    if os.environ.get("OPENAI_API_KEY"):
        example_comprehensive_error_handling()
        example_batch_and_file_errors()
    
    print("=" * 50)
    print("Error Handling Summary")
    print("=" * 50)
    print("""
Exception Hierarchy:
├── AuthenticationError     - Invalid/missing API key
├── RateLimitError          - Too many requests (has retry_after)
├── InvalidRequestError     - Bad parameters
│   └── ContextLengthExceededError - Input too long
├── ContentFilterError      - Blocked by safety filters
├── APIError                - Generic API errors (has status_code)
├── BatchError              - Batch processing failures
├── FileError               - File operation failures
├── ProviderNotSupportedError - Unknown provider
└── ProviderImportError     - Missing provider package

Best Practices:
1. Always wrap API calls in try/except
2. Handle RateLimitError with exponential backoff
3. Check retry_after for rate limits
4. Log errors with full context for debugging
5. Distinguish retriable vs non-retriable errors
""")


if __name__ == "__main__":
    main()
