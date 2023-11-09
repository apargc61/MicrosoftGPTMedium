# pip install spacy
# python -m spacy download en_core_web_sm
import spacy

class IntentMatcher:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        # Define some sample intents and their representative sentences
        self.intents = {
            "greeting": self.nlp("hello"),
            "goodbye": self.nlp("goodbye"),
            "ask_weather": self.nlp("weather today"),
            "ask_time": self.nlp("current time")
        }

    def predict_intent(self, text):
        input_vector = self.nlp(text)
        max_similarity = -1
        matched_intent = None

        # Calculate cosine similarity for each intent and pick the highest
        for intent, intent_vector in self.intents.items():
            similarity = input_vector.similarity(intent_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_intent = intent

        return matched_intent

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
