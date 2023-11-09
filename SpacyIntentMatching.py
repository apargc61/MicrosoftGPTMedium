# pip install spacy
# python -m spacy download en_core_web_sm

import spacy
from spacy.util import minibatch
import random

class IntentMatcher:
    def __init__(self):
        self.nlp = spacy.blank('en')
        self.textcat = self.nlp.create_pipe("textcat")
        self.nlp.add_pipe(self.textcat, last=True)

        # Define some sample training data
        self.train_data = [
            ("hello", {"cats": {"greeting": 1.0, "goodbye": 0.0, "ask_weather": 0.0, "ask_time": 0.0}}),
            ("hi there", {"cats": {"greeting": 1.0, "goodbye": 0.0, "ask_weather": 0.0, "ask_time": 0.0}}),
            ("bye", {"cats": {"greeting": 0.0, "goodbye": 1.0, "ask_weather": 0.0, "ask_time": 0.0}}),
            ("what's the weather", {"cats": {"greeting": 0.0, "goodbye": 0.0, "ask_weather": 1.0, "ask_time": 0.0}}),
            ("what time is it", {"cats": {"greeting": 0.0, "goodbye": 0.0, "ask_weather": 0.0, "ask_time": 1.0}})
        ]

        # Add labels to the text classifier
        for _, annotations in self.train_data:
            for label, _ in annotations["cats"].items():
                self.textcat.add_label(label)

        self.train()

    def train(self, iterations=20):
        # Training the model
        optimizer = self.nlp.begin_training()
        for i in range(iterations):
            random.shuffle(self.train_data)
            losses = {}
            batches = minibatch(self.train_data, size=8)
            for batch in batches:
                texts, annotations = zip(*batch)
                self.nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)

    def predict_intent(self, text):
        doc = self.nlp(text)
        intent = max(doc.cats, key=doc.cats.get)
        return intent

def main():
    matcher = IntentMatcher()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        intent = matcher.predict_intent(user_input)
        print(f"Detected Intent: {intent}")

if __name__ == "__main__":
    main()
