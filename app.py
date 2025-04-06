import requests

import streamlit as st
import pandas as pd
from rag_pipeline import generate_shl_assessment
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
