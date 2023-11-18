import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    print("Getting pdf text.")
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    print("Getting text chunks.")
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    print("Getting embeddings.")
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    print("Creating vector store.")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    print("Getting conversation chain.")
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    # store and display chat history
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate[st.session_state.chat_history]:
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )


def main():
    # load environment variables
    load_dotenv()

    # configure Streamlit page
    st.set_page_config(
        page_title="Chat with the Data Stories Archive", page_icon=":books:"
    )
    st.header("Chat with Data Stories :books:")

    # handle user input
    user_question = st.text_input("Ask a question about housing, property or planning:")
    if user_question:
        handle_userinput(user_question)

    # add css styling
    st.write(css, unsafe_allow_html=True)

    # initialise session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # create sidebar
    with st.sidebar:
        st.subheader("Your documents")
        # store user submitted documents
        pdf_docs = st.file_uploader(
            "Upload your documents here and click on ''Process'",
            accept_multiple_files=True,
        )
        # process documents
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create embeddings and populate vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain and store session state to persist variables
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
