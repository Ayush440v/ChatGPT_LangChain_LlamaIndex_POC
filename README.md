# chatgpt--langchain-retrieval

Simple script to use ChatGPT on your own data.

Here we are using Text based information stored in a text file.

## Installation

Install [Langchain](https://github.com/hwchase17/langchain) and other required packages.
```
pip install langchain openai chromadb tiktoken unstructured
```
Modify `constants.py.default` to use your own [OpenAI API key](https://platform.openai.com/account/api-keys), and rename it to `constants.py`.

Place your own data into `data/data.txt`.

## Example usage
Test reading `data/data.txt` file.
```

Test it on termina by running

> python chatgpt.py

or send the prompt directly with

> python chatgpt.py "what is my manager's name"
Your manager's name is Shanal.
```