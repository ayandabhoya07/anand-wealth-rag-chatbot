"""
gemini_client.py  -  the only file that talks to Google Gemini.

AI_Bot.py uses just two functions from here:
    embed(texts, task_type)  ->  turns text into numbers (vectors) so we can search
    chat(messages, ...)      ->  asks Gemini to write a reply
"""
import os
import time

import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from a local .env file, if there is one

# ---------------------------------------------------------------
# 1. SETTINGS (change these if you want a different model)
# ---------------------------------------------------------------
EMBED_MODEL = "gemini-embedding-001"   # model that makes the vectors
LLM_MODEL = "gemini-2.5-flash"         # model that writes the answers
EMBED_DIM = 768                        # length of each vector
BATCH_SIZE = 50                        # how many texts we embed per request


# ---------------------------------------------------------------
# 2. CONNECT TO GEMINI
# ---------------------------------------------------------------
def get_api_key():
    """On Streamlit Cloud the key comes from Secrets; on your PC from the .env file."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


@st.cache_resource  # create the connection only once
def get_client():
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in Streamlit Secrets.")
    return genai.Client(api_key=api_key)


def call_with_retry(request, tries=4):
    """Run a Gemini request. If it fails (e.g. too many requests), wait and try again."""
    for attempt in range(tries):
        try:
            return request()
        except Exception as error:
            if attempt == tries - 1:   # last try failed -> give up
                raise
            wait_seconds = 5 * (2 ** attempt)   # waits 5s, 10s, 20s
            print(f"Gemini error: {error}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


# ---------------------------------------------------------------
# 3. EMBEDDINGS  (text -> vectors)
# ---------------------------------------------------------------
def embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    """
    texts     : a list of strings
    task_type : "RETRIEVAL_DOCUMENT" for knowledge-base chunks,
                "RETRIEVAL_QUERY"    for the user's question
    returns   : a numpy array with one vector per text
    """
    client = get_client()
    all_vectors = []

    # Send the texts in small groups so one request is never too big
    for start in range(0, len(texts), BATCH_SIZE):
        group = texts[start:start + BATCH_SIZE]

        result = call_with_retry(lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=group,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBED_DIM,
            ),
        ))

        for item in result.embeddings:
            all_vectors.append(item.values)

    vectors = np.array(all_vectors, dtype=np.float32)

    # Scale every vector to length 1 so similarity scores are consistent
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (lengths + 1e-10)


# ---------------------------------------------------------------
# 4. CHAT  (messages -> reply text)
# ---------------------------------------------------------------
def chat(messages, temperature=0.3, max_tokens=500):
    """
    messages    : list like [{"role": "system", "content": "..."},
                             {"role": "user", "content": "..."},
                             {"role": "assistant", "content": "..."}]
    temperature : 0 = very predictable, higher = more creative
    max_tokens  : maximum length of the reply
    returns     : the reply as a string
    """
    client = get_client()

    # Gemini takes the "system" message separately from the conversation
    system_text = ""
    conversation = []
    for message in messages:
        if message["role"] == "system":
            system_text += message["content"] + "\n"
        else:
            # Gemini calls the assistant "model"
            role = "model" if message["role"] == "assistant" else "user"
            conversation.append(
                types.Content(role=role, parts=[types.Part(text=message["content"])])
            )

    settings = types.GenerateContentConfig(
        system_instruction=system_text or None,
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),  # no "thinking", so the whole budget goes to the answer
    )

    response = call_with_retry(lambda: client.models.generate_content(
        model=LLM_MODEL,
        contents=conversation,
        config=settings,
    ))

    return (response.text or "").strip()