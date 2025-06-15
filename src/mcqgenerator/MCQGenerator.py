import json
import pandas as pd
import traceback
import PyPDF2
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv() 

KEY=os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
print("Base Path :", BASE_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
print("Base Path :", PROJECT_ROOT)

file_path = os.path.join(PROJECT_ROOT, "Response.json")
print("File Path : ",file_path)

with open(file_path, "r") as f:
    RESPONSE_JSON = json.load(f)

TEMPLATE = """
You are an expert at designing multiple choice questions (MCQs).

Given the **text** below, your task is to generate **{number} unique MCQs** for students studying **{subject}**. The quiz should be written in a **{tone} tone**.

Make sure:
- All questions are relevant to the provided text
- No questions are repeated
- Each question has 4 answer choices (A–D)
- Exactly one correct answer is marked
- All questions are clear and unambiguous

---
### TEXT:
{text}
---

### REQUIRED FORMAT (response_json):
{response_json}

Please strictly follow the JSON structure above and ensure it contains exactly {number} MCQs.
"""


quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
    )

quiz_chain= LLMChain(
    llm=llm, 
    prompt=quiz_generation_prompt, 
    output_key="quiz", 
    verbose=True
    )

TEMPLATE2 = """
You are an expert English grammarian and question designer.

Your task is to review the following Multiple Choice Quiz intended for {subject} students. Perform the following:

1. **Complexity Analysis**  
   - Evaluate the overall language complexity and cognitive demand of each question.  
   - Use no more than **50 words** in total for the complexity analysis.  

2. **Suitability Check**  
   - Determine whether the questions are appropriate for the cognitive and analytical level of {subject} students.  
   - If any questions are **too simple, too complex, or grammatically incorrect**, rewrite them accordingly.  
   - Maintain clarity and pedagogical soundness.  

---
### Original Quiz
{quiz}
---

### Output Format

Please return the following:
- **Complexity_Analysis**: A short summary (≤ 50 words).
- **Updated_Quiz**: A revised version of the MCQs if needed, in the same structure as the input.
- **Suggestions**: Optional brief notes on why any changes were made.

Start your response below:
"""


quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"], 
    template=TEMPLATE2
    )

review_chain = LLMChain(
    llm=llm, 
    prompt=quiz_evaluation_prompt, 
    output_key="review", 
    verbose=True
    )

generate_evaluation_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["review", "quiz"],
    verbose=True
    )