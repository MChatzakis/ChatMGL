import argparse
import json
import os

from zipfile import ZipFile
from generative_model import GenerativeModel
from tqdm import tqdm

import numpy as np
import random
import torch

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Filter our some deprecation warnings
import warnings
warnings.filterwarnings("ignore")

GENERATION_TOKENS = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_unzip(path):
    """Validates the model path:
        1. If the path leads to a zipped model, it unzips it and returns the path to the unzipped model
        2. If the path leads to a folder, it returns the path to the folder of the model
            * Does not handle cases where the folder contains wrong files
        3. If the path is not absolute, it changes it to absolute

    Args:
        path (_type_): path to ChatMGL model

    Returns:
        string: Absolute path to the ChatMGL model
    """
    model_path = None
    if path.endswith(".zip"):
        print(f"Model is zipped. Unzipping {path}")
        with ZipFile(path, "r") as zObject:
            model_folder = path.replace(".zip", "")
            zObject.extractall(path=model_folder)

        print(f"Model unzipped in {model_folder}. Loading the model from there")
        model_path = model_folder      
    else:
        print(f"Model is not zipped. Loading the model directly from {path}")
        model_path = path
    
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
        print(f"Model path is not absolute. Changing it to {model_path}")
    
    return model_path


def answer_prompts(input_questions: str, output_filename: str, model):
    """
    Generates answers to the questions in the input file and saves them in the output file.
    """
    answers = []
    prompts = json.load(open(input_questions, "r"))
    for entry in tqdm(prompts, desc="ChatMGL Generation"):
        guid = entry["guid"]
        question = entry["question"]
        if "choices" in entry.keys() and entry["choices"] is not None:
            choices = entry["choices"]
            question = question + " " + " ".join(choices)
        
        answer = model.generate(question, max_new_tokens=GENERATION_TOKENS)
        answers.append({"guid": guid, "model_answer": answer})

    json.dump(answers, open(output_filename, "w"), indent=4)
    print(f"Total {len(answers)} Answers saved in {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_questions_path",
        type=str,
        default="../prompts.json",
        help="Path to the data file containing the questions",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="../answers_chatMGL.json",
        help="Filename of the output json file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../answers_chatMGL.json",
        help="Path to the ChatMGL model",
    )
    parser.add_argument(
        "--generation_tokens",
        type=str,
        default=50,
        help="Max token number to be generated by the model",
    )

    args = parser.parse_args()

    print("\n==== ChatMGL - Generative Model Script ====\n\n")
    print("-Before starting, please download ChatMGL from Google Drive: https://drive.google.com/drive/folders/1Klcx6gJHiJIj-BS6vB2-OM9Gx49qyxPd")
    print("-Alternatively, you can download ChatMGL using the get_chatMGL.sh script")
    print("-We also provide a tutorial on how to use ChatMGL, available in ChatMGL.md")
    
    path = check_unzip(args.model_path) 
    
    print("\nLoading model...")
    model = GenerativeModel(path, device=DEVICE)
    print("Model loaded successfully")
    
    GENERATION_TOKENS = int(args.generation_tokens)
    print(f"Model generation tokens are set to {GENERATION_TOKENS}. The value greatly affects the generation time.")

    print("\nGenerating answers...")
    answer_prompts(args.input_questions_path, args.output_filename, model)
