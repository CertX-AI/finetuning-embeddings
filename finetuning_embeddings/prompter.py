"""Prompt functions for the data generator and data augmentator module."""

##########################   DATA GENERATOR MODULE  ##########################
##########################   NEW QUESTION PROMPTS   ##########################
##########################          SYSTEM          ##########################
##########################          PROMPTS         ##########################

def generate_system_question_prompt() -> str:
    """Return a strict, detailed system prompt for generating exactly ONE.

    alternate question ("Alternate Instruction") whose answer is the
    supplied Answer.

    Returns:
       str: A formatted string containing the system prompt.
    """
    return """You are an expert standards editor.

GOAL
Generate ONE new question ("Alternate Instruction") such that the provided Answer would still be fully correct and complete.

ABSOLUTE RULES
1. Stay on the SAME TOPIC as the source question.
2. The new question must be FULLY ANSWERED by the given Answer alone
   — no extra info required.
3. The new question must be ENTIRELY SUPPORTED by the supplied Context.
4. Produce EXACTLY ONE line of output, in this format (no markdown, no list bullets, no quotes):
   Output: <your alternate question here>

FORMAT EXAMPLE (for illustration only)
Output: What systematic RAMS management process does EN 50126-1 establish?

COMMON MISTAKES TO AVOID
x Adding explanations, bullet points, or multiple lines.
x Quoting or restating the Answer.
x Changing 'Output:' to something else.
x Introducing information not justified by the Context.

Follow the rules precisely. Deviations will be rejected.
""".strip() # noqa: E501



##########################          SYSTEM          ##########################
##########################          PROMPTS         ##########################


##########################      USER  PROMPTS       ##########################
##########################        ZERO SHOT         ##########################

def generate_user_question_prompt(
    instruction: str,
    answer: str,
    context: str,
    topic: str,
    notes: str,
) -> str:
    """Build the detailed *user* prompt that instructs the model to output.

    exactly ONE "Alternate Instruction" (new question) such that the
    provided Answer is still fully correct.

    Parameters
    ----------
    instruction : str
        The original question from the dataset.
    answer : str
        The existing answer that must remain valid.
    context : str
        The supporting excerpt from the source document.
    topic : str
        The overarching subject area.
    notes : str | None
        Optional extra notes.

    Returns:
    -------
    str
        A fully formatted prompt ready to send as the user message.
    """
    return f"""
You have received one record containing five fields extracted from the official document of EN 50126 which is a European standard that specifies and demonstrates the reliability, availability, maintainability, and safety (RAMS) of railway applications:

Original Instruction: {instruction}
Answer: {answer}
Context: {context}
Topic: {topic}
Notes: {notes}

YOUR TASK
Generate ONE new question — called the "Alternate Instruction" — for which the **exact Answer above** remains completely correct and sufficient.

REQUIREMENTS
• The new question must stay on the same Topic.
• It must be answerable solely with the given Answer (no extra info).
• It must be fully supported by the Context text.
• It must NOT be the same as, nor a trivial re-phrase of, the Original Instruction.

OUTPUT FORMAT
Return **exactly one line**, with no additional text or markup:

Output: <your alternate question here>

EXAMPLE (for illustration only — do NOT copy):
Output: What systematic RAMS management process does EN 50126-1 establish?

PROHIBITED
x Additional lines, bullet points, or markdown.
x Repeating or quoting the Answer.
x Using any prefix other than "Output:".

Follow these rules precisely. A deviation will be rejected.
""".strip() # noqa: E501

##########################      USER  PROMPTS       ##########################
##########################        ZERO SHOT         ##########################
