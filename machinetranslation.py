from transformers import MarianMTModel, MarianTokenizer

# Load the pretrained MarianMT model for English to French translation
model_name = "Helsinki-NLP/opus-mt-en-fr"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Define the input text in English
input_text = "Hello, how are you?"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate translation
translation = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

# Decode the translated text
translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

# Print the translation
print("English input: ", input_text)
print("French translation: ", translated_text)
