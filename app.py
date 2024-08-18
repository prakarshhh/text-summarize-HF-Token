import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint
import time

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def summarize_content():
    try:
        with st.spinner("Waiting..."):
            if "youtube.com" in generic_url:
                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
            else:
                loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                               headers={"User-Agent": "Mozilla/5.0"})
            docs = loader.load()
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            output_summary = chain.run(docs)
            st.success(output_summary)
    except Exception as e:
        if "429" in str(e):
            st.error("Rate limit reached. Please try again later.")
        else:
            st.exception(f"Exception: {e}")

if st.button("Summarize the Content from YT or Website"):
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or a website URL")
    else:
        summarize_content()
