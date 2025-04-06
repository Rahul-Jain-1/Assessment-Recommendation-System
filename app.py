import requests
import streamlit as st
import pandas as pd

def streamlit_app():
    st.title("SHL Assessment Recommendation System")
    st.write("Enter a query....")
    user_input = st.text_area("Query", height=150)
    
    if st.button("Get Recommendations"):
        if user_input:
            try:
                with st.spinner("Generating recommendations..."):
                    url = "https://dao5id03ig.execute-api.us-east-2.amazonaws.com/devel/recommendation-system"
                    params = {"query": user_input}
                    response = requests.get(url, params=params)
                    response.raise_for_status()  # Raises an exception for HTTP errors
                    
                    # Parse the JSON response and extract the "data" field
                    api_response = response.json()
                    data = api_response.get("data", [])
                    
                    # Display recommendations if any data is returned
                    if data:
                        df = pd.DataFrame(data)
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
