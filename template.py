import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name  = "TextSummerizer"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__int__.py",
    f"src/{project_name}/components/__int__.py",
    f"src/{project_name}/utils/__int__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging//__int__.py",
    f"src/{project_name}/config/__int__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__int__.py",
    f"src/{project_name}/entity/__int__.py",
    f"src/{project_name}/constants/__int__.py",
    "config/config.ymal",
    "params.ymal",
    "app.py",
    "main.py",
    "Dockerfile",
    "reqiurements.txt",
    "setup.py",
    "research/trials.ipynb",
]  

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")