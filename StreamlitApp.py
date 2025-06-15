import os
import sys
import json
import traceback
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from mcqgenerator.utils import read_file, get_table_data
from mcqgenerator.MCQGenerator import generate_evaluation_chain
from mcqgenerator.logger import logging
    #mcqgenerator

with open("/Users/arjun/Desktop/MCQ Generator/experiment/Response.json", "r") as file: 
    RESPONSE_JSON = json.load(file)

st.title("MCG GENERATOR APPLICATION")

with st.form("User input"):

    uploaded_file = st.file_uploader("upload pdf or text")

    mcq_count = st.number_input("Number of MCQs", min_value=5, max_value = 25)
    
    subject = st.text_input("Subject", max_chars=20)

    tone = st.text_input("Complexty level of Question", max_chars=20, placeholder="simple")

    button = st.form_submit_button("Generat MCQs")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text=read_file(uploaded_file)
                #Count tokens and the cost of API call
                with get_openai_callback() as cb:
                    response= generate_evaluation_chain(
                        {
                        "text": text,
                        "number": mcq_count,
                        "subject":subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                            }
                    )
                #st.write(response)

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                
                if isinstance(response, dict):
                    #Extract the quiz data from the response
                    quiz=response.get("quiz", None)
                    if quiz is not None:
                        table_data=get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)
                            #Display the review in atext box as well
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the table data")

                else:
                    st.write(response)