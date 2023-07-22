from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium')

# Load text data from webpage
with open('webpage_text.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

# Tokenize the text data
tokenized_data = tokenizer.encode(text_data, add_special_tokens=False, truncation=True)

# Create a TextDataset
dataset = TextDataset(tokenized_data, block_size=128)

# Create a DataCollator for Language Modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define the TrainingArguments
training_args = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    logging_dir='./logs',
    evaluation_strategy="steps",
    eval_steps=500,
    dataloader_num_workers=4
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Fine-tune the model on the text data
trainer.train()

# Generate answers using the fine-tuned model
model.eval()

# Example conversation to generate an answer
user_input = "What is the capital of France?"

# Encode the user input
input_ids = tokenizer.encode(user_input, add_special_tokens=True, return_tensors="pt")

# Generate response
output = model.generate(input_ids=input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)

# Decode and print the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Answer:", response)
