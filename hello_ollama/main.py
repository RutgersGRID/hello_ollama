# ollama_cli.py

import click
from langchain_community.llms import Ollama

@click.command()
@click.option('--model', default='phi', help='Name of the LLM model to use (default: gemma2)')
@click.argument('input_text', type=str)
def main(model, input_text):
    # Initialize the Ollama model
    llm = Ollama(model=model)
    
    # Invoke the model with the input text
    response = llm.invoke(input_text)
    
    # Print the response
    click.echo(response)

if __name__ == '__main__':
    main()
