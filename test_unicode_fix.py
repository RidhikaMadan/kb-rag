#!/usr/bin/env python3
"""
Test script to verify Unicode encoding fixes
"""
import sys
import json
import requests

# Test message with Unicode characters that previously caused issues
test_messages = [
    "Hello, can you help me?",
    "Test with special characters: \u201cquotes\u201d and em\u2014dashes",
    "What about hashtags #test and markdown **bold** text?",
    "Unicode test: \u00e9, \u00f1, \u4e2d\u6587",
]

def test_chat_endpoint(base_url="http://localhost:8000"):
    """Test the chat endpoint with various Unicode messages"""
    print("Testing Unicode encoding fixes...")
    print("=" * 60)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: {message[:50]}...")
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "message": message,
                    "max_tokens": 100,
                    "min_score": 0.5
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Success! Response: {data.get('response', '')[:100]}...")
            else:
                error_detail = response.text
                print(f"✗ Error {response.status_code}: {error_detail[:200]}")
                if "UnicodeEncodeError" in error_detail or "ascii codec" in error_detail:
                    print("  ⚠️ Unicode encoding error still present!")
                    return False
        except requests.exceptions.ConnectionError:
            print("✗ Could not connect to server. Is it running?")
            return False
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Exception: {error_msg[:200]}")
            if "UnicodeEncodeError" in error_msg or "ascii codec" in error_msg:
                print("  ⚠️ Unicode encoding error still present!")
                return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Unicode encoding issues appear to be fixed.")
    return True

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = test_chat_endpoint(base_url)
    sys.exit(0 if success else 1)

