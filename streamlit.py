from main import ChatBot
import streamlit as st

# Initialize the chatbot class
bot = ChatBot()

# Streamlit page settings
st.set_page_config(page_title="Health and Wellness Tips Chatbot")

# Sidebar configuration
with st.sidebar:
    st.title('Health and Wellness Chatbot')
    st.write('Ask for tips on healthy living, fitness, nutrition, and more!')

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result

# Initialize session state for storing chat messages
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me anything about health and wellness."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input box for user prompt
if input := st.chat_input("Ask for health and wellness advice here..."):
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Fetching wellness advice..."):
            response = generate_response(input)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)