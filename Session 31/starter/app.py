import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from support_bot.chatbot import build_chain

from support_bot.config import MODEL_NAME

st.set_page_config(page_title="Acme Support", page_icon="🎧")
st.title("🎧 Acme Support Assistant")


@st.cache_resource
def get_chain():
    return build_chain()


chain = get_chain()

# 1. Create the transcript once per browser session.
if "history" not in st.session_state:
    st.session_state.history = []

# 2. Re-draw the whole transcript on every run.
for message in st.session_state.history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.text)

# 3. Handle a new message.
if user_text := st.chat_input("How can I help?"):
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        reply = st.write_stream(
            chain.stream({
                "input": user_text,
                "history": st.session_state.history,
            })
        )

    # 4. Append AFTER the call, so the model never sees the current turn twice.
    st.session_state.history.append(HumanMessage(user_text))
    st.session_state.history.append(AIMessage(reply))

with st.sidebar:
    st.subheader("Session")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption(f"Turns stored: {len(st.session_state.get('history', [])) // 2}")
    if st.button("🔄 New conversation", width="stretch"):
        st.session_state.history = []
        st.rerun()