# -*- coding: utf-8 -*-
"""
Complete Azure ML pipeline (SDK v2) script.
Save this script as `azure_ml_pipeline_toy.py` and run it in an environment 
where you are authenticated with Azure (e.g., via Azure CLI `az login`).
"""

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, dsl, command

# 1. Initialize ML Client
# Replace these placeholder values with your actual Azure resource details
subscription_id = "YOUR_SUBSCRIPTION_ID"
resource_group = "YOUR_RESOURCE_GROUP"
workspace_name = "YOUR_WORKSPACE_NAME"

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# 2. Define Components
prep_component = command(
    name="prep_data_component",
    display_name="Prepare Data",
    description="A component to clean raw data",
    inputs={"data": Input(type="uri_folder")},
    outputs={"clean_data": Output(type="uri_folder")},
    code=".",
    command="echo 'Preparing data...' && mkdir -p ${{outputs.clean_data}} && cp -r ${{inputs.data}}/* ${{outputs.clean_data}}/ || true",
    environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
)

train_component = command(
    name="train_model_component",
    display_name="Train Model",
    description="A component to train a model",
    inputs={"clean_data": Input(type="uri_folder")},
    outputs={"model": Output(type="uri_folder")},
    code=".",
    command="echo 'Training model...' && mkdir -p ${{outputs.model}} && echo 'my_dummy_model' > ${{outputs.model}}/model.txt",
    environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
)

# 3. Define Pipeline
@dsl.pipeline(
    name="toy-layered-training-pipeline",
    description="Prepare data, then train a toy model.",
    default_compute="cpu-cluster",
)
def training_pipeline(raw_data):
    # Layer 1: prepare the data
    prep = prep_component(data=raw_data)

    # Layer 2: train using Layer 1's output
    train = train_component(
        clean_data=prep.outputs.clean_data
    )

    # Return the final model artifact
    return {"model": train.outputs.model}

# 4. Instantiate and Submit Job
# Use an existing data asset on your Workspace (e.g., bank-data version 1)
pipeline_job = training_pipeline(
    raw_data=Input(
        type="uri_folder",
        path="azureml:bank-data:1"
    )
)

try:
    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="toy-layered-training",
    )
    print("Pipeline submitted successfully!")
    print("Job name:", submitted_job.name)
    print("Studio URL:", submitted_job.studio_url)
except Exception as e:
    print("Error submitting job (Verify your credentials and Azure resources):")
    print(e)
