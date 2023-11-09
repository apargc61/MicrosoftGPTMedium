import openai
from transformers import T5ForConditionalGeneration, T5Tokenizer
# from opensearchpy import OpenSearch
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
# from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.pipelines import ExtractiveQAPipeline

# Set up OpenAI GPT-4 API
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key

# Set up the Elasticsearch Document Store
document_store = ElasticsearchDocumentStore(
    host="localhost", port=9200, index="your_index_name"
)  # Replace with your OpenSearch index details

# Set up the Elasticsearch Retriever
retriever = ElasticsearchRetriever(document_store=document_store)

# Set up the T5-base model
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Set up the ExtractiveQAPipeline
pipeline = ExtractiveQAPipeline(reader=None, retriever=retriever)

# Create a JSONL file for storing results
output_file = "document_summaries.jsonl"

# Retrieve documents from the index and generate summaries
document_results = []

with open(output_file, "w") as jsonl_file:
    documents = document_store.get_all_documents()
    for document in documents:
        document_text = document.text

        # Generate a GPT-4 summary
        gpt4_summary = openai.Completion.create(
            engine="davinci", prompt=document_text, max_tokens=100
        ).choices[0].text

        # Generate a T5 summary
        input_text = "summarize: " + document_text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        t5_summary_ids = model.generate(inputs.input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
        t5_summary = tokenizer.decode(t5_summary_ids[0], skip_special_tokens=True)

        # Store results in the JSONL file
        result = {
            "original_document": document_text,
            "gpt4_summary": gpt4_summary,
            "t5_summary": t5_summary,
        }
        jsonl_file.write(json.dumps(result) + "\n")

print(f"Summaries saved to {output_file}")
