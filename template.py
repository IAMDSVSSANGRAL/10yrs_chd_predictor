# import os
# from pathlib import Path
# import logging

# logging.basicConfig(level=logging.INFO)

# project_name="predictor"

# list_of_files=[
#     f"src/{project_name}/__init__.py",
#     f"src/{project_name}/components/__init__.py",
#     f"src/{project_name}/components/data_ingestion.py",
#     f"src/{project_name}/components/data_transformation.py",
#     f"src/{project_name}/components/model_tranier.py",
#     f"src/{project_name}/components/model_monitering.py",
#     f"src/{project_name}/pipelines/__init__.py",
#     f"src/{project_name}/pipelines/training_pipeline.py",
#     f"src/{project_name}/pipelines/prediction_pipeline.py",
#     f"src/{project_name}/exception.py",
#     f"src/{project_name}/logger.py",
#     f"src/{project_name}/utils.py",
#     "main.py",
#     "app.py",
#     "Dockerfile",
#     "requirements.txt",
#     "setup.py",
#     ".env",
#     "artifacts/"
# ]

# for filepath in list_of_files:
#     filepath = Path(filepath)
#     filedir, filename = os.path.split(filepath)

#     if filedir != "":
#         os.makedirs(filedir, exist_ok=True)
#         logging.info(f"Creating directory:{filedir} for the file {filename}")

    
#     if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
#         with open(filepath,'w') as f:
#             pass
#             logging.info(f"Creating empty file: {filepath}")


    
#     else:
#         logging.info(f"{filename} is already exists")


import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define project name
project_name = "predictor"

# List of required files and folders
list_of_items = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",  # Fixed typo: 'tranier' -> 'trainer'
    f"src/{project_name}/components/model_monitoring.py",  # Fixed typo: 'monitering' -> 'monitoring'
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "notebooks/data/",
    "notebooks/EDA.ipynb",
    "notebooks/model_training.ipynb",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    ".env",
    "artifacts/",  # Correctly handle folder creation
]

# Iterate through each path
for item in list_of_items:
    path = Path(item)

    # If item is a directory (ends with '/')
    if item.endswith("/"):
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"✅ Created directory: {path}")

    # If item is a file
    else:
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the parent directory exists
        if not path.exists():
            path.touch()  # Create an empty file
            logging.info(f"📄 Created file: {path}")
        else:
            logging.info(f"⚠️ File already exists: {path}")
