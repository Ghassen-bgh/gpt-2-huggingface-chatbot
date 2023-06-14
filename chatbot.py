# chatbot.py
from response import get_response

def chat():
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        response = get_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    chat()
