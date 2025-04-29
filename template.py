import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    ".env",
    "setup.py",
    "src/prompt.py",
    "app.py",
    "research/trials.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating new file: {filepath}")
    elif os.path.getsize(filepath) == 0:
        logging.warning(f"File {filepath} is empty, but not overwriting it.")
    else:
        logging.info(f"{filename} already exists and is not empty.")


requirements_path = Path("requirements.txt")
if not requirements_path.exists():
    with open(requirements_path, "w") as f:
        f.write("-e git+https://github.com/akashbisht004/Med-bot.git#egg=Generative_ai_project\n")
    logging.info("Created default requirements.txt with Git dependency.")
else:
    logging.info("requirements.txt already exists.")
