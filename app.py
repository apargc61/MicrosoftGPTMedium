#imports
from flask import Flask, render_template, request
# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__,template_folder='templates')
# #create chatbot
# englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = ChatterBotCorpusTrainer(englishBot)
# trainer.train("chatterbot.corpus.english") #train the chatter bot for english

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(user_input):
    bot_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_output_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(bot_output_ids[0], skip_special_tokens=True)
    return bot_response


#define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    response = generate_response(userText)
    new_response = response[len(userText):]
    return new_response #generate_response(userText)
    #return str(englishBot.get_response(userText))

if __name__ == "__main__":
    app.run()