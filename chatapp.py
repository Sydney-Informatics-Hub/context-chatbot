# Chatbot model for OpenAI ChatGPT chatbot using context-based vector embeddings

"""
This is a chatbot that uses OpenAI's GPT-3 model to answer questions. 
It uses a pre-trained model to generate the embeddings for the document sections, 
and then uses cosine similarity to find the most relevant sections for a given question.

The document sections are stored in a CSV file, and the embeddings are stored in a CSV file.
Prompts are constructed by concatenating the most relevant sections, and then passing the prompt 
to the OpenAI completion API.

Tokens are counted using the TikTok tokenizer.

For more details see OpenAI's documentation on Embeddings:
https://platform.openai.com/docs/guides/embeddings

This code has been designed to create the chatbot for the Geodata-Harvester project.
A webapp implementation for this chatbot is available at:
https://sydney-informatics-hub.github.io/geodata-harvester/

Author: Sebastian Haan
"""

import openai
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity
import os
import numpy as np
import pandas as pd

### Settings for OpenAI and Chatbot ####

CHATBOT_NAME = "Geodata-Harvester Chatbot"

# Filename for OpenAI API key
# (CREATE FILE AND ADD YOUR OWN KEY THERE - see https://platform.openai.com/docs/quickstart)
FNAME_API_KEY = "openai_api_key.txt"

SEPARATOR = "\n* "
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
name_encoding = encoding.name

separator_len = len(encoding.encode(SEPARATOR))

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

MAX_SECTION_LEN = 800
MAX_COMPLETION_TOKENS = 800

FNAME_EMBEDDINGS = "embeddings/dfembeddings.csv"
FNAME_SECTIONS = "embeddings/sections.csv"

COMPLETIONS_API_PARAMS = {
    # Use temperature of 0.0 to get the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": MAX_COMPLETION_TOKENS ,
    "model": COMPLETIONS_MODEL,
}


#### Functions for Chat ####

def count_tokens(string: str, encoding_name: str = 'cl100k_base') -> int:
    """Returns the number of tokens in a text string.

    INPUT:
        string: str, the text to count tokens for
        encoding_name: str, the name of the encoding to use

    OUTPUT:
        num_tokens: int, the number of tokens in the string
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    INPUT:
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "tokens", "0", "1", ... up to the length of the embedding vectors.

    OUTPUT:
    A dictionary mapping (title, heading, tokens) to the embedding vector.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading" and c != "tokens"])
    return {
           (r.title, r.heading, r.tokens): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    """
    Returns the vector embedding for the supplied text.

    INPUT:
        text: str, the text to embed
        model: str, the OpenAI model to use for embedding

    OUTPUT:
        result_embedding: list[float], the embedding vector
    """
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str, int), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.

    INPUT:
        query: str, the question to answer
        contexts: dict[(str, str), np.array], the embeddings of the document sections

    OUTPUT:
        document_similarities: list[(float, (str, str))], the list of document sections, sorted by relevance in descending order
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(
    question: str, 
    context_embeddings: dict, 
    df: pd.DataFrame,
    show_sections: bool = False
    ) -> str:
    """
    Construct the prompt for the OpenAI completion API.

    INPUT:
        question: str, the question to answer
        context_embeddings: dict[(str, str), np.array], the embeddings of the document sections
        df: pd.DataFrame, the dataframe containing the document sections
        show_sections: bool, whether to show the selected document sections in the prompt

    OUTPUT:
        prompt: str, the prompt to pass to the OpenAI completion API
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.   
        #print('section_index', section_index)    
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    if show_sections:
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))
    

    # Construct the prompt header and include any rules for the answer:
    header = """Answer the question as truthfully as possible using the provided context below. 
    If the answer seems to be a list, provide answer in form of bullet points.
    If the answer is not contained within the text below, say 
    "Hmm, it seems this information is not available for the Geodata-Harvester."
    ###
    \n\n Context:\n
    ###
    """

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str, str), np.array],
    show_prompt: bool = False
    ) -> str:
    """
    Answer a question using the context of the document.

    INPUT:
        query: str, the question to answer
        df: pd.DataFrame, the dataframe containing the document sections
        document_embeddings: dict[(str, str, str), np.array], the embeddings of the document sections
        show_prompt: bool, whether to print the prompt to the console

    OUTPUT:
        response: str, the answer to the question
    """
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    return response["choices"][0]["text"] #.strip(" \n")


def query(
    question: str, 
    df: pd.DataFrame, 
    context_embeddings: dict[(str, str, str), np.array]
    ) -> str:
    """
    Answer a question using the context of the document.

    INPUT:
        question: str, the question to answer
        df: pd.DataFrame, the dataframe containing the document sections
        context_embeddings: dict[(str, str, str), np.array], the embeddings of the context sections

    OUTPUT:
        response: str, the answer to the question
        tokens_used: int, the estimated number of tokens used to answer the question
    """
    # Count the number of tokens used for embedding the question
    tokens_question = count_tokens(question)

    # Construct the prompt and count the number of tokens used for prompt
    prompt = construct_prompt(question, context_embeddings, df)
    token_prompts = count_tokens(prompt)

    # Answer the question and count the number of tokens used for the response
    response = answer_query_with_context(prompt, df, context_embeddings)
    tokens_response = count_tokens(response)

    # Count the total number of tokens used
    tokens_used = token_prompts + tokens_response + tokens_question

    return response, tokens_used


def main():
    """
    Main function to run the app. 
    Note that this is just a command-line demo app. For a web app, please use any of the web frameworks.
    """
    print("\U0001F916------------------------------------------------------------\U0001F916")
    print(f"Welcome to the {CHATBOT_NAME}!")
    print("This app uses context-based OpenAI's GPT-3 to answer questions.")

    # initialise API from file containing key
    try:   
        with open(FNAME_API_KEY, 'r') as f:
            openai.api_key = f.read().strip()
        # Check if valid key
        openai.Engine.list()
    except:
        print(f"Error: OpenAI key not found. Please add your key to the file {FNAME_API_KEY}.")
        return

    print("Type 'quit' to exit.")

    # Load the embeddings and the document sections
    # check if files exist
    if not os.path.isfile(FNAME_EMBEDDINGS):
        print(f"Error: {FNAME_EMBEDDINGS} not found.")
        return
    if not os.path.isfile(FNAME_SECTIONS):
        print(f"Error: {FNAME_SECTIONS} not found.")
        return
    context_embeddings = load_embeddings(FNAME_EMBEDDINGS)
    df = pd.read_csv(FNAME_SECTIONS, header=0)
    # Set the index to be (document title, heading, tokens)
    df.set_index(list([df.title, df.heading, df.tokens]), inplace=True)

    while True:
        question = input("Q: ")
        if question == "quit":
            break
        answer, tokens_used = query(question, df, context_embeddings)
        print(f"A: {answer} ({tokens_used} tokens used)")


if __name__ == "__main__":
    main()


# Write README.md for this project
"""
# Context-based OpenAI GPT-3 Chatbot

This is a demo app that uses OpenAI's GPT-3 to answer questions using the context of multiple documents.

For a webapp demonstration, please see [here](https://sydney-informatics-hub.github.io/geodata-harvester/_includes/embed_chat.html)


## Requirements

The following Python dependencies are required:

- `openai`
- `pandas`
- `numpy`
- `tiktoken`

## How to use

1. Create an OpenAI account and get an API key.
2. Add your API key to the file `openai_api_key.txt`.
3. Install the dependencies using `pip install -r requirements.txt`.
4. Run `chatapp.py` to start the app.

## How it works

This app is based on the documents from the [Geodata-Harvester project](https://github.com/Sydney-Informatics-Hub/geodata-harvester)
and uses the OpenAI's GPT-3 to answer questions using the context of the document.
The context and embeddings have been pre-processed and are stored in the folder `embeddings` as CSV files.

The app uses the following main steps to answer a question:

1. Find the most relevant document sections for a given question by using the embeddings of the document sections.
2. Construct a prompt for the question using the most relevant document sections.
3. Use the prompt to answer the question using OpenAI's GPT-3.

The number of tokens used to answer the question is estimated by counting the number of tokens used for the question, the prompt and the response.


## References

- [Geodata-Harvester Chatbot Webapp and Documentation](https://sydney-informatics-hub.github.io/geodata-harvester/)
- [Geodata-Harvester Github](https://github.com/Sydney-Informatics-Hub/geodata-harvester)
- [OpenAI GPT-3 API](https://platform.openai.com/docs/introduction)
- [OpenAI GPT-3 embeddings](https://platform.openai.com/docs/guides/embeddings)
- [OpenAI GPT-3](https://openai.com/blog/gpt-3-apps/)

## License

This open-source project is licensed under the LGPL License - see the [LICENSE](LICENSE) file for details.

@Copyright 2023 Sebastian Haan



"""


