import requests

import streamlit as st
import pandas as pd

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rag_pipeline import RAG_pipeline
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class internal_Sources(BaseModel):
    url: str = Field(description="Url that contains job description.")
   
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")


   
def scrape_text(url):
    """
    Scrapes the page at the given URL and returns its text content,
    removing all HTML tags.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.212 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve page: Status code {response.status_code}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text

def generate_shl_assessment(input_text):
    """
    Scrapes the target URL, extracts its text, then passes a prompt along with the text
    to the Google LLM (Gemini model integration, if available) to generate a refined job description.
    """
    # Step 1: Scrape the text from the URL
    structured_llm = llm.with_structured_output(internal_Sources)
    result = structured_llm.invoke(f"Extract url from this text :: {input_text}") 
    # check if there is url or not 
    extracted_text = ""
    if result is not None:
        url = result.url
        print(f"url is {url}")
        extracted_text = scrape_text(url)
        
    prompt = (
        "You are an expert in building good queries for intelligent recommendation systems for hiring assessments. "
        "Generate a concise and clear nlp search query that captures the essential skills, requirements required for given job role.."
        "Start query with Give SHL assesment related to. If time given include that maximum duration time is .."
        "Include important keywords from query only. Dont include extra words."
        "This query will be used to query a RAG engine for recommending relevant SHL assessments from our catalog.\n\n"
        f"User original query: {input_text}\n\n"
    )
    
    if extracted_text:
        prompt += f"Extracted text:\n{extracted_text}\n"

    response = llm.invoke(prompt)
    
    refined_prompt = response.content
    print(f"query is {refined_prompt}")
    
    # object for calling rag pipeline
    rag = RAG_pipeline()
    result = rag.query_rag(query=refined_prompt)
    print(f"result is {result}")
    return result

# user_input ="Here is a   JD text : https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in , can you recommend some assessment that can help me screen applications. Time limit is less than 30 minutes"

def streamlit_app():
    st.title("SHL Assessment Recommendation System")
    st.write("Enter a query....")
    user_input =  st.text_area("Query", height=150)
        # Text area for user input.
    
    if st.button("Get Recommendations"):
            if user_input:
                try:
                    with st.spinner("Generating recommendations..."):
                        
                        result = generate_shl_assessment(user_input)
                        
                        # Assume result is a list of dictionaries.
                        if isinstance(result, list) and result:
                            df = pd.DataFrame(result)
                            st.write("Recommendations:")
                            st.dataframe(df)
                        else:
                            st.write("No recommendations found.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a query or URL.")

if __name__ == "__main__":
    streamlit_app()
