from pipelines.embed_pipeline import embed_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

import click

@click.command()
@click.option(
    '--embed_model_name', default='sentence-transformers/all-MiniLM-L6-v2', help='Embed model name',
)

def main(embed_model_name: str):
    """Main function"""
    # get mlflow tracking uri
    mlflow_tracking_uri = get_tracking_uri()
    print(f"MLflow tracking uri: {mlflow_tracking_uri}")
    embed_pipeline(embed_model_name)

if __name__ == '__main__':
    main()
