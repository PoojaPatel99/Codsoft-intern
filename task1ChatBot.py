import re

# Define the chatbot's responses based on predefined rules
def chatbot_response(user_input):
    # Normalize input to lowercase
    user_input = user_input.lower()

    # Rules for basic pattern matching
    if re.search(r'hello|hi|hey', user_input):
        return "Hello! How can I help you today?"

    elif re.search(r'how are you', user_input):
        return "I'm just a bot, but I'm doing great! How about you?"

    elif re.search(r'what is your name', user_input):
        return "I am a simple chatbot. You can call me Chatbot!"

    elif re.search(r'bye|goodbye', user_input):
        return "Goodbye! Have a great day!"

    elif re.search(r'what can you do', user_input):
        return "I can answer your questions and help with basic tasks. Just ask me anything!"

    elif re.search(r'help', user_input):
        return "Sure! Ask me something like 'How are you?' or 'What's your name?' and I'll try to respond."

    elif re.search(r'your favorite color', user_input):
        return "I don't have a favorite color, but I think blue is nice!"

    else:
        return "I'm sorry, I didn't understand that. Can you rephrase?"

# Main loop to keep the chatbot running
def chat():
    print("Chatbot: Hello! Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ")
        
        # If the user types 'bye', exit the loop
        if 'bye' in user_input.lower():
            print("Chatbot: Goodbye! Take care!")
            break
        
        response = chatbot_response(user_input)
        print("Chatbot:", response)

# Run the chatbot
if __name__ == "__main__":
    chat()
