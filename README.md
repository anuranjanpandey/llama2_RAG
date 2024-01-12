![Python 3.11.7](https://img.shields.io/badge/python-3.11.7-blue.svg)

```bash
pip install zenml["server"]
zenml up
```


```bash
zenml integration install mlflow -y
``` 



```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```