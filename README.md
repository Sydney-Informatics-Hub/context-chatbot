# Context-based OpenAI GPT-3 Chatbot

This is a demo app that uses OpenAI's GPT-3 to answer questions using context specific documents. 

The context for this chatbot is derived from the [Geodata-Harvester documentation](https://sydney-informatics-hub.github.io/geodata-harvester/) and the webapp is embedded in the same page (see Section "What is it").

<center><img title="Webapp preview" alt="Webapp preview image" src="docs/images/preview_webapp.png" width="500" /></center>


## How to use

1. Download or fork the repository.
2. Create an OpenAI account and get an API key.
3. Create the file `openai_api_key.txt` and add the OpenAI API key to the file.
4. Install the dependencies using `pip install -r requirements.txt`.
5. Run `chatapp.py` to start the app:
    ```bash
    python chatapp.py
    ```

## How it works

This app is based on the documents from the [Geodata-Harvester project](https://github.com/Sydney-Informatics-Hub/geodata-harvester) and uses the OpenAI's GPT-3 to answer questions regarding the context of the document.
The context and vector embeddings have been pre-processed and are stored in the folder `embeddings` as CSV files.

The app uses the following main steps to answer a question:

1. Find the most relevant document sections for a given question by using the embeddings of the document sections.
2. Construct a prompt for the question using the most relevant document sections (based on cosine similarity).
3. Add guidelines to the prompt to ensure that the answer is relevant to the context of the document.
4. Use the prompt to answer the question using OpenAI's GPT-3.
5. Return the answer and the number of tokens used to answer the question.

The total number of tokens used to answer the question is estimated by counting the sum of number of tokens used for the question embedding, the prompt and the response.

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


## Known Limitations

- The app is limited to provide only answers that are related to the context of the documents.
- Some of the training data is not updated and might not be relevant anymore.
- The app is currently limited to 1000 tokens per request.
- Embeddings and matching content need to be provided.
- The app is not optimized for speed and might be slow for large datasets.
- The app does not provide any feedback on the quality of the answers.


## References

- [Geodata-Harvester Chatbot Webapp and Documentation](https://sydney-informatics-hub.github.io/geodata-harvester/)
- [Geodata-Harvester Github](https://github.com/Sydney-Informatics-Hub/geodata-harvester)
- [OpenAI GPT-3 API](https://platform.openai.com/docs/introduction)
- [OpenAI GPT-3 embeddings](https://platform.openai.com/docs/guides/embeddings)

## License

This open-source project is licensed under the GNU LESSER GENERAL PUBLIC LICENSE v2.1. See the [LICENSE](LICENSE) file for details.

@Copyright 2023 Sebastian Haan