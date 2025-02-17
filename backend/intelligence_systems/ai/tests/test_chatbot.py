from import_manager import *


async def test_sebastian():
    config = ChatConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.8,
        max_tokens=150,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1
    )
    
    sebastian = Chatbot(config=config)
    
    test_inputs = [
        "Sebastian, make me some tea.",
        "What do you think of humans?",
        "Show me your true form.",
        "This is an order!",
        "Are you really just a butler?"
    ]
    
    print("=== Sebastian Quantum AI Test ===\n")
    for input_text in test_inputs:
        response = await sebastian.generate_response(input_text)
        print(f"Master: {input_text}")
        print(f"Sebastian: {response}\n")
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_sebastian())
