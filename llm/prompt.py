from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

# System message to set the role of the model
system_message = """
You are a code similarity evaluator. Given a coding question, a candidate's answer, and an AI-generated answer, your job is to determine how similar the two answers are. 
The similarity score must be any floating point number within the range 0 and 1, where:

- 0 means completely different
- 1 means exactly the same

Your response should ONLY contain the similarity score, nothing else.
"""

# Human message for similarity detection
human_message = """
Coding Question: {question}
Candidate's Answer: {candidate_answer}
AI-Generated Answer: {ai_answer}

Based on the above, what is the similarity score between the candidate's answer and the AI-generated answer? Return only a floating point number between 0 and 1.
"""


# Define the system message template (to specify the role of the model)
system_message_template = SystemMessagePromptTemplate.from_template(
    system_message)

# Define the human message prompt template (which takes input variables)
human_message_template = HumanMessagePromptTemplate.from_template(
    human_message)

# Combine both the system and human messages in one prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_template, human_message_template])
