from flask import Flask
from flask_cors import CORS
# from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load pretrained T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('mrm8488/bert-tiny-5-finetuned-squadv2')
model = AutoModelForQuestionAnswering.from_pretrained('mrm8488/bert-tiny-5-finetuned-squadv2')

# Define questions and context
questions = [
    "What is Gen Garage?",
    "What is the purpose of Gen Garage?",
    "Who can join Gen Garage?",
    "How can I be a mentor for Gen Garage?",
    "What is Gen Z Campus?",
    "What all locations is Gen Garage community present in?",
    "In which domains does Gen Garage develop projects?",
    "When was Gen Garage started?",
    "What Daily Dose Engagements does Gen Garage provide?",
    "What are the Clubs of Gen Garage?"
]

context = ""<b>Welcome to Gen Garage!<br><br> I'm your friendly GENIE, at your service!<br><br>  Here are some options for you to explore the website and know about it:<br></b><b>*</b> About Gen Garage<br><b>*</b> Missions of Gen Garage<br><b>*</b> Leads of Gen Garage<br><b>*</b> Current Projects<br><b>*</b> Achievements"

app = Flask("__name__")
CORS(app)


@app.route('/input/<text>')
def send_output(text):
    # input_text = f"question: {text} context: {context}"
    # input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)['input_ids']
    # with torch.no_grad():
    #     outputs = model.generate(input_ids, max_length=50)
    #     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    inputs = tokenizer(text, context, add_special_tokens=True, return_tensors="pt")
 
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
 
    # Get the most likely answer span
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1  # Add 1 because slicing in Python doesn't include the end index
 
    answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])
 
    # Print the question and answer
    # print(f"Question: {question}")
    # print(f"Answer: {answer}")
    return answer


if __name__ == "__main__":
    app.run(debug=True)
