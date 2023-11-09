class IntentMatcher:
    def __init__(self):
        # Define keywords for each intent
        self.intents = {
            "greeting": ["hello", "hi", "hey"],
            "goodbye": ["bye", "goodbye", "see you"],
            "ask_weather": ["weather", "forecast", "sunny", "rain"],
            "ask_time": ["time", "clock", "hour"]
        }

    def match_intent(self, user_input):
        # Normalize the input (convert to lowercase)
        user_input = user_input.lower()
        
        # Check each intent for keyword matches
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in user_input:
                    return intent  # Return the matched intent
        return "unknown"  # Return unknown if no match found

def main():
    matcher = IntentMatcher()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        intent = matcher.match_intent(user_input)
        print(f"Detected Intent: {intent}")

if __name__ == "__main__":
    main()
