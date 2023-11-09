from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

# Create a chatbot instance
chatbot = ChatBot('CustomerSupportBot')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot on English language data
trainer.train('chatterbot.corpus.english')

# Create a list of custom responses for the chatbot
custom_responses = [
    'How can I track my order?',
    'You can track your order by logging into your account and visiting the order history page.',
    'What is your return policy?',
    'Our return policy allows returns within 30 days of purchase. Please visit our returns page for more details.',
]

# Train the chatbot on custom responses
custom_trainer = ListTrainer(chatbot)
custom_trainer.train(custom_responses)

# Chat with the chatbot
print("CustomerSupportBot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = chatbot.get_response(user_input)
    print("CustomerSupportBot:", response)
