from flask import Flask
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load pretrained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

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

context = '''
Gen Garage was founded in Mumbai in 2019 as an innovative platform that unites fresh graduates, interns, mentors, and 
industry leaders. It focuses on systematically transforming talents for industrial settings and includes all Capgemini 
employees by default. For further engagement, individuals can connect with business line leads or the core team via the 
'meet the team' page. Additionally, Gen Z Campus, a collaborative learning initiative with I&D Gen Z Arena, offers 
immersive learning opportunities, mentorship, and an innovators' space. Operating nationwide, Gen Garage has a 
significant presence in Bangalore, Mumbai, Pune, Chennai, Hyderabad, Kolkata, and Delhi NCR. It specializes in Data 
Engineering, Data Science, Machine Learning, Data Visualization, and Cloud Technologies. Formerly known as Millennial 
Garage, it underwent rebranding in 2023 to 'Gen Garage' with a federated model. Weekly activities include Knowledge 
Sharing Sessions on Monday, Tuesday Tips, Innovation of the Week on Wednesday, Guru Gyan on Thursday, and Masti Friday. 
The community features six clubs: Mantri Club, Art Lens Club, Social Media Champs, Escape Club, Magic Pen, and Falcon 
Club
'''

app = Flask("__name__")
CORS(app)


@app.route('/input/<text>')
def send_output(text):
    input_text = f"question: {text} context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)['input_ids']
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


if __name__ == "__main__":
    app.run(debug=True)
