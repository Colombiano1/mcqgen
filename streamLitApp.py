import os
import json
import traceback
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.mcqgenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

with open('/Users/colombianoamoussougbo/Desktop/CAREER/LEARNINGS/GenAI/Assignments/mcqgen/response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

#creating a title for the app   
st.title("mcq creation with langchain")

#creatinga form using st.form
with st.form("user_inputs"):
    #file Upload
    uploaded_file=st.file_uploader("upload a pdf or txt file")
    
    #input fields
    mcq_count=st.number_input("no of mcq", min_value=3, max_value=50)
    
    #subject
    subject=st.text_input("insert subject", max_chars=20)
    
    #quiz tone
    tone=st.text_input("complexity Level Of Question", max_chars=20, placeholder="simple")
    
    #add button
    button = st.form_submit_button("create MCQs")
    
    #check if the button is clicked anf all fields have inputs
    if button and uploaded_file is not None and mcq_count and subject and tone:
        print("running MCQ") 
        with st.spinner("loading..."):
            
            try:
                text=read_file(uploaded_file)
                #count token and cost of api call
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                        {
                            "text": text,
                            "number":mcq_count,
                            "subject":subject,
                            "tone":tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
                #st.write(response)
                
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("ERROR")
                
            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response, dict):
                    #extract the quiz data from the response
                    quiz=response.get("quiz", None)
                    if quiz is not None:
                        table_data=get_table_data(quiz)
                        if table_data is not None and isinstance(table_data, list):
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)
                        else:
                            st.error("ERROR IN THE TABLE DATA")
                    else:
                        st.write(response)
                        
