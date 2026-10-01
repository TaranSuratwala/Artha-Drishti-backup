import os
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.DEBUG)

from AgentEngine import ArthaAgent

class MockComponent:
    pass

def main():
    try:
        agent = ArthaAgent(screener=MockComponent(), predictor=MockComponent(), pipeline=MockComponent())
        print("Agent initialized")
        
        print("Streaming response for: 'hello'")
        for chunk in agent.stream("hello", session_id="test_session"):
            print("CHUNK:", chunk)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
