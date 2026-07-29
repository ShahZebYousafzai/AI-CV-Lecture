from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .llm import get_model
from .prompts import SUPPORT_SYSTEM_PROMPT


def build_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPPORT_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    return prompt | get_model() | StrOutputParser()