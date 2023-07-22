from transformers import pipeline
# other large language models to use: https://huggingface.co/blog/llama2
# Use GPU if available, otherwise use CPU
# device = -1 if not torch.cuda.is_available() else 0
# model_name = "distilbert-base-cased" 
# model_name = "MvpForQuestionAnswering"
# model_name = "meta-llama/Llama-2-7b-chat-hf" this does not work
# model_name = "tiiuae/falcon-7b"
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # this was the best
# Create a Question Answering pipeline using the Hugging Face library
qa_pipeline = pipeline(task="question-answering", model = model_name)


# context = "Paris is the capital and largest city of France."
# Load text data from webpage
with open('webpage_text.txt', 'r', encoding='utf-8') as f:
    context = f.read()

# Example question and context
question = "Please tell me about Nepal"

# Perform question-answering using the GPU/CPU
result = qa_pipeline(question=question, context=context)

# Print the answer
print("Answer:", result["answer"])


