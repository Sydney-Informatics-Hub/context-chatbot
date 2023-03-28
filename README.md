# Context-based OpenAI GPT-3 Chatbot

This is a demo app that uses OpenAI's GPT-3 to answer questions using context specific documents. 

For the webapp demonstration, please see [here](https://sydney-informatics-hub.github.io/geodata-harvester/_includes/embed_chat.html)


## How to use

1. Create an OpenAI account and get an API key.
2. Add your API key to the file `openai_api_key.txt`.
3. Install the dependencies using `pip install -r requirements.txt`.
4. Run `chatapp.py` to start the app:
    ```bash
    python chatapp.py
    ```

## How it works

This app is based on the documents from the [Geodata-Harvester project](https://github.com/Sydney-Informatics-Hub/geodata-harvester) and uses the OpenAI's GPT-3 to answer questions using the context of the document.
The context and embeddings have been pre-processed and are stored in the folder `embeddings` as CSV files.

The app uses the following main steps to answer a question:

1. Find the most relevant document sections for a given question by using the embeddings of the document sections.
2. Construct a prompt for the question using the most relevant document sections.
3. Use the prompt to answer the question using OpenAI's GPT-3.

The number of tokens used to answer the question is estimated by counting the number of tokens used for the question, the prompt and the response.

## Requirements

The following main Python dependencies are required:

- `openai`
- `pandas`
- `numpy`
- `tiktoken`
- `matplotlib`
- `plotly`
- `scikit-learn`
- `scipy`

For a full list of dependencies, please see the file `requirements.txt`.
The app has been tested with Python 3.9 and 3.10.


## References

- [Geodata-Harvester Chatbot Webapp and Documentation](https://sydney-informatics-hub.github.io/geodata-harvester/)
- [Geodata-Harvester Github](https://github.com/Sydney-Informatics-Hub/geodata-harvester)
- [OpenAI GPT-3 API](https://platform.openai.com/docs/introduction)
- [OpenAI GPT-3 embeddings](https://platform.openai.com/docs/guides/embeddings)
- [OpenAI GPT-3](https://openai.com/blog/gpt-3-apps/)

## License

This open-source project is licensed under the Lesser General Public License (LGPL) v3.0. See the [LICENSE](LICENSE) file for details.

@Copyright 2023 Sebastian Haan