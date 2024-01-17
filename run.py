from pipelines.index_builder import docs_to_index_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

import click

@click.command()
@click.option(
    '--embed_model_name', default='sentence-transformers/all-MiniLM-L6-v2', help='Embed model name',
)

# to do update seperate functions for Initializing huggingface embedding pipeline, Building the vector index, Ingesting data into the vector index, Initializing the Huggingface pipeline, and Initializing the Retrival QA chain
def main():
    """Main function"""
    # get mlflow tracking uri
    mlflow_tracking_uri = get_tracking_uri()
    print(f"MLflow tracking uri: {mlflow_tracking_uri}")
    docs_to_index_pipeline()

if __name__ == '__main__':
    main()

