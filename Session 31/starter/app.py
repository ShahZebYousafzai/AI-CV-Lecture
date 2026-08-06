import uuid

import streamlit as st

from support_bot.agent import build_agent
from support_bot.config import MODEL_NAME

st.set_page_config(page_title="Acme Support", page_icon="🎧")
st.title("🎧 Acme Support Assistant")


@st.cache_resource
def get_agent():
    return build_agent()

agent = get_agent()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "transcript" not in st.session_state:
    st.session_state.transcript = []

config = {"configurable": {"thread_id": st.session_state.thread_id}}

with st.sidebar:
    st.subheader("Session")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}`")
    if st.button("🔄 New conversation", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.transcript = []
        st.rerun()

for role, text in st.session_state.transcript:
    with st.chat_message(role):
        st.markdown(text)

if user_text := st.chat_input("How can I help?"):
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.transcript.append(("user", user_text))

    with st.chat_message("assistant"):
        with st.status("Working…", expanded=True) as status:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_text}]},
                config,
            )
            # Only inspect messages produced by THIS turn.
            messages = result["messages"]
            start = max(i for i, m in enumerate(messages) if m.type == "human")
            for message in messages[start:]:
                for call in getattr(message, "tool_calls", None) or []:
                    st.write(f"🔧 `{call['name']}` ← `{call['args']}`")
            status.update(label="Done", state="complete", expanded=False)

        reply = result.get("structured_response")
        if reply is None:                       # nothing structured came back
            answer = messages[-1].text
            st.markdown(answer)
        else:
            answer = reply.answer
            st.markdown(answer)
            st.caption(
                f"`{reply.category}` · sentiment `{reply.sentiment}` · "
                f"needs_human `{reply.needs_human}` · order `{reply.order_id}`"
            )
        st.session_state.transcript.append(("assistant", answer))