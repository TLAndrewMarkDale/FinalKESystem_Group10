import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import re
from langchain_core.output_parsers import StrOutputParser
from streamlit_float import *
# Streamlit UI
# The UI is divided into two columns.
# The left column is for the user to select the language and upload the document.
# The right column is for the user to interact with the assistant.
# The assistant can answer questions about the document in the selected language.
st.set_page_config(page_title="Doc Lingo", page_icon="🌍", layout="wide")
top_left, top_center, top_right = st.columns([1, 2, 1])
with top_left:
    align_left,_,_ = st.columns([1, 1, 1])
    with align_left:
        st.image("image_assets/lang_world.png", width=80)
with top_center:
    _,align_center,_ = st.columns([1, 2, 1])
    with align_center:
        st.title("Doc Lingo")
with top_right:
    _,_,align_right = st.columns([1, 1, 1])
    with align_right:
        st.image("image_assets/document.png", width=80)

# The prompt template for the assistant to answer questions about the document.
prompt_template_qa = """
The user wants to know more about the document provided by {context}
Please answer the user in their specified language {language}
Answer in another language only if explicitly asked.
The user's question is delimited by triple backticks.
Input: ```{input}```
"""

# The prompt template for the assistant to update the text in the application.
PROMPT = PromptTemplate.from_template(prompt_template_qa)


# Languages that can be used with ChatGPT.
# The key is the language in the language's own language.
# The value is the language in English.
# The languages are in alphabetical order.
# I wanted to have this in a separate file, but it was causing a frozen_importlib error.
# More can be read at: https://seo.ai/blog/how-many-languages-does-chatgpt-support

dict_of_languages = {
    "Shqip": "Albanian",
    "العربية": "Arabic",
    "Հայերեն": "Armenian",
    "अवधी": "Awadhi",
    "Azərbaycanca": "Azerbaijani",
    "Башҡортса": "Bashkir",
    "Euskara": "Basque",
    "Беларуская": "Belarusian",
    "বাংলা": "Bengali",
    "भोजपुरी": "Bhojpuri",
    "Bosanski": "Bosnian",
    "Português do Brasil": "Brazilian Portuguese",
    "Български": "Bulgarian",
    "廣州話": "Cantonese (Yue)",
    "Català": "Catalan",
    "छत्तीसगढ़ी": "Chhattisgarhi",
    "中文": "Chinese",
    "Hrvatski": "Croatian",
    "Čeština": "Czech",
    "Dansk": "Danish",
    "डोगरी": "Dogri",
    "Nederlands": "Dutch",
    "English": "English",
    "Eesti": "Estonian",
    "Føroyskt": "Faroese",
    "Suomi": "Finnish",
    "Français": "French",
    "Galego": "Galician",
    "ქართული": "Georgian",
    "Deutsch": "German",
    "Ελληνικά": "Greek",
    "ગુજરાતી": "Gujarati",
    "हरियाणवी": "Haryanvi",
    "हिन्दी": "Hindi",
    "Magyar": "Hungarian",
    "Bahasa Indonesia": "Indonesian",
    "Gaeilge": "Irish",
    "Italiano": "Italian",
    "日本語": "Japanese",
    "Javanese": "Javanese",
    "ಕನ್ನಡ": "Kannada",
    "कश्मीरी": "Kashmiri",
    "Қазақша": "Kazakh",
    "कोंकणी": "Konkani",
    "한국어": "Korean",
    "Кыргызча": "Kyrgyz",
    "Latviešu": "Latvian",
    "Lietuvių": "Lithuanian",
    "Македонски": "Macedonian",
    "मैथिली": "Maithili",
    "Bahasa Melayu": "Malay",
    "Malti": "Maltese",
    "官话": "Mandarin",
    "官話": "Mandarin Chinese",
    "मराठी": "Marathi",
    "मारवाड़ी": "Marwari",
    "闽南语": "Min Nan",
    "Moldovenească": "Moldovan",
    "Монгол хэл": "Mongolian",
    "Crnogorski": "Montenegrin",
    "नेपाली": "Nepali",
    "Norsk": "Norwegian",
    "ଓଡ଼ିଆ": "Oriya",
    "پښتو": "Pashto",
    "فارسی": "Persian (Farsi)",
    "Polski": "Polish",
    "Português": "Portuguese",
    "ਪੰਜਾਬੀ": "Punjabi",
    "राजस्थानी": "Rajasthani",
    "Română": "Romanian",
    "Русский": "Russian",
    "संस्कृतम्": "Sanskrit",
    "Santali": "Santali",
    "Српски": "Serbian",
    "سنڌي": "Sindhi",
    "සිංහල": "Sinhala",
    "Slovenčina": "Slovak",
    "Slovenščina": "Slovene",
    "Українська": "Ukrainian",
    "اردو": "Urdu",
    "Ўзбекча": "Uzbek",
    "Tiếng Việt": "Vietnamese",
    "Cymraeg": "Welsh",
    "吳語": "Wu"
}

# The OpenAI API key is stored in an environment variable.
model = 'gpt-3.5-turbo'
key = st.secrets['OPENAI_API_KEY']

# The prompt template to use the LLM to update streamlit components
update_text_prompt = """
        Translate: {text} 
        To the following language: {language}
        """

# The function to handle the document upload.
# The function takes the split pages of the document as input.
# The function sets the chat_disabled and download_disabled to False.
# This is the RAG component of the application.
def handle_docupload(pages) -> None:
    st.session_state.chat_disabled = False
    st.session_state.download_disabled = False
    prompt = PromptTemplate.from_template(prompt_template_qa)
    llm = ChatOpenAI(api_key=key)
    embeddings = OpenAIEmbeddings(api_key=key)
    persist_directory = 'persist_chroma'
    vectordb = Chroma.from_documents(pages, 
                                         embedding=embeddings, 
                                         persist_directory=persist_directory)
    retriever = vectordb.as_retriever()
    combine_docs_chain = create_stuff_documents_chain(llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever,
                                             combine_docs_chain)
    st.session_state.retrieval_chain = retrieval_chain


# The function to handle the document upload.
# The function takes the uploaded document as input.
# The function writes the uploaded document to a temporary directory.
# The function loads and splits the document.
# The function returns the split pages of the document.
def doc_upload(uploaded_doc):
    with open(f"tempDir/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
    if uploaded_file.name.endswith(".pdf"):
        pages = PyPDFLoader(f"tempDir/{uploaded_file.name}").load_and_split()
    else: 
        pages = TextLoader(f"tempDir/{uploaded_file.name}").load_and_split()
    return pages

# The function to handle the translation of the document.
# The function takes the split pages of the document as input.
# The function translates the document to the selected language.
# The function sets the translation_ready to True.
# The function sets the data to the translated text.
# The function uses the LLM to translate the document.
def handle_translation(pages) -> None:
    st.session_state.data = None
    with st.spinner("Translating document..."):
        text = ""
        prompt = PromptTemplate.from_template(update_text_prompt)
        llm = ChatOpenAI(model=st.session_state.openai_model, openai_api_key=key)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        for page in pages:
            response = llm_chain.invoke({'text': page.page_content, 'language': st.session_state.language})
            response = response['text']
            if response is not None:
                text += response
        st.session_state.translation_ready = True
        st.session_state.data = text

# The function to update the text in the application.
# The function takes the new language as input.
# The function updates the interface with the new language.
def update_application_text() -> None:
    with st.spinner("Updating interface..."):
        prompt = PromptTemplate.from_template(update_text_prompt)
        to_update = [st.session_state.chat_placeholder, st.session_state.subtitle]
        llm = ChatOpenAI(model=st.session_state.openai_model, openai_api_key=key)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.batch([{'text': to_update[0], 'language': st.session_state.language}, 
                                    {'text': to_update[1], 'language': st.session_state.language}])
        st.session_state.chat_placeholder = response[0]['text']
        st.session_state.subtitle = response[1]['text']

if 'chat_disabled' not in st.session_state:
    st.session_state.chat_disabled = True
if 'download_disabled' not in st.session_state:
    st.session_state.download_disabled = True
if 'translation_ready' not in st.session_state:
    st.session_state.translation_ready = False
if 'button_label' not in st.session_state:
    st.session_state.button_label = "Update"
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None


llm = ChatOpenAI(model=model,
                 api_key=key,
                 temperature=0)
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = model

if "messages" not in st.session_state:
    st.session_state["messages"] = []


if 'subtitle' not in st.session_state:
    st.session_state.subtitle = "Ask questions about a document in any of the supported languages"
if 'chat_placeholder' not in st.session_state:
    st.session_state.chat_placeholder = "Type your question here..."
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'data' not in st.session_state:
    st.session_state.data = None


left, right = st.columns([1, 1])
with left:
    choice = st.selectbox(label="Language", 
                          options=dict_of_languages.keys())
    st.session_state.language = choice
    if st.button(label=st.session_state.button_label): update_application_text()
    uploaded_file = st.file_uploader("Upload a document", 
                                     type=['txt', 'pdf'])
    if uploaded_file is None:
        st.session_state.chat_disabled = True
        st.session_state.download_disabled = True
        st.session_state.data = None
        st.session_state.translation_ready = False
        st.session_state.retrieval_chain = None
    else:
        pages = doc_upload(uploaded_file)
        translate, download, _ = st.columns([1, 1, 1])
        handle_docupload(pages)
        with translate:
            st.button("Translate", on_click=handle_translation, args=(pages,))
        with download:
            if st.session_state.data is not None:
                st.download_button(label="Download", 
                                   data=st.session_state.data, 
                                   mime="text/plain", file_name=f"{uploaded_file.name}_translated_{st.session_state.language}.txt", 
                                   disabled=not st.session_state.translation_ready)


if prompt := st.chat_input(st.session_state.chat_placeholder, disabled=st.session_state.chat_disabled):
    #st.chat_message('user').write(prompt)
    result = st.session_state.retrieval_chain.invoke({'input': prompt, 'language': st.session_state.language, 
                                                    "context": st.session_state.messages})
    #st.chat_message('assistant').write(result['answer'])
    st.session_state.messages.append((prompt, result['answer']))

with right:
    with st.container(height=500):
        st.caption(st.session_state.subtitle)
        for message in st.session_state.messages:
            st.chat_message('user').write(message[0])
            st.chat_message('assistant').write(message[1])




    

