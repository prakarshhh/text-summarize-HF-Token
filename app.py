import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint
import time

# Set page configuration with a custom theme
st.set_page_config(
    page_title="🧠 AI Summary Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for animations and styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f8f9fa;
        }

        h1, h2, h3 {
            color: #007bff;
        }

        .stButton button {
            background-color: #007bff;
            color: white;
            font-size: 18px;
            padding: 12px;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }

        .stTextInput > div > input {
            border: 2px solid #007bff;
            padding: 10px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }

        .stTextInput > div > input:focus {
            border-color: #0056b3;
            box-shadow: 0 0 10px rgba(0, 91, 187, 0.2);
        }

        .stSlider > div > div {
            color: #007bff;
        }

        .stSidebar > div {
            background-color: #e9ecef;
        }

        .stSidebar h1, .stSidebar h2 {
            color: #007bff;
        }

        .stSidebar .stSelectbox > div > div {
            border: 2px solid #007bff;
            border-radius: 5px;
            transition: all 0.3s ease;
        }

        .stSidebar .stSelectbox > div > div:focus {
            border-color: #0056b3;
        }

        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #6c757d;
        }

        .footer a {
            color: #007bff;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subheader with enhanced styling
st.title("🚀 AI Summary Generator")
st.subheader("Transforming YouTube Videos and Websites into Concise Summaries")

# Sidebar for API key and options with icons
with st.sidebar:
    st.write("### 🔐 API Settings")
    hf_api_key = st.text_input("Huggingface API Token", value="", type="password")

    st.write("### ⚙️ Summary Options")
    language = st.selectbox("🌐 Select Language", ["English", "Spanish", "French", "German", "Hindi"])
    word_count = st.slider("📝 Summary Length (in words)", min_value=50, max_value=500, value=300, step=50)

    with st.expander("🔧 Advanced Settings"):
        max_length = st.slider("Max Length", min_value=50, max_value=2000, value=150, step=50)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# Main input for URL with animation
generic_url = st.text_input("🔗 Enter the YouTube or Website URL", placeholder="Paste URL here...", label_visibility="visible")

# Language-specific repo IDs
language_repo_mapping = {
    "English": "mistralai/Mistral-7B-Instruct-v0.3",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Hindi": "Helsinki-NLP/opus-mt-en-hi"
}

repo_id = language_repo_mapping.get(language, "mistralai/Mistral-7B-Instruct-v0.3")
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=max_length, temperature=temperature, token=hf_api_key)

# Prompt template
prompt_template = f"""
Provide a summary of the following content in {word_count} words:
Content: {{text}}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarize content function
def summarize_content():
    try:
        with st.spinner("⚙️ Summarizing... Please wait..."):
            # Determine loader type based on URL
            if "youtube.com" in generic_url:
                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
            else:
                loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                               headers={"User-Agent": "Mozilla/5.0"})
            # Load documents
            docs = loader.load()
            
            # Create the summary chain and run it
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            output_summary = chain.run(docs)
            st.success("✅ Summary generated successfully!")
            st.write(output_summary)
    except Exception as e:
        if "429" in str(e):
            st.error("Rate limit reached. Please try again later.")
        else:
            st.error(f"An error occurred: {str(e)}")

# Button to trigger summary generation with hover effect
if st.button("💡 Summarize the Content"):
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and the URL to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
    else:
        summarize_content()

# Footer with additional info
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Powered by <a href="https://www.langchain.com" target="_blank">LangChain</a> and <a href="https://huggingface.co" target="_blank">HuggingFace</a>. 
        Customize your summaries with the power of AI. Adjust the settings in the sidebar to tailor the output to your needs.
    </div>
    """,
    unsafe_allow_html=True
)
