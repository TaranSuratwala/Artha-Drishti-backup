import os
from dotenv import load_dotenv
load_dotenv()

from AgentEngine import ArthaAgent

class MockComponent:
    pass

def main():
    agent = ArthaAgent(screener=MockComponent(), predictor=MockComponent(), pipeline=MockComponent())
    print("Agent initialized")
    
    app = agent.graph.compile()
    
    print("Testing stream_mode='messages'")
    try:
        for msg, metadata in app.stream({"messages": [("user", "Explain Piotroski strategy")]}, {"configurable": {"thread_id": "3"}}, stream_mode="messages"):
            if msg.type == "AIMessageChunk" or msg.type == "ai":
                if isinstance(msg.content, str) and msg.content:
                    print("CONTENT:", msg.content)
                elif isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, dict) and "text" in part:
                            print("CONTENT_PART:", part["text"])
                if hasattr(msg, 'tool_call_chunks') and msg.tool_call_chunks:
                    for tc in msg.tool_call_chunks:
                        if tc.get("name"):
                            print("TOOL:", tc["name"])
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    main()
