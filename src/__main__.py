import sys
import json
import argparse
from pathlib import Path
from pydantic import ValidationError
from src.models import FunctionDefinition, PromptInput
from llm_sdk import Small_LLM_Model

def load_json_file(file_path: str):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def main():

    #1. This part tells the program which flags to look for
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()

    #2. Load and validate function definitions
    raw_functions = load_json_file(args.functions_definition)
    try:
        functions = [FunctionDefinition(**fn) for fn in raw_functions]
        print(f"Successfully loaded {len(functions)} function definitions.")
    except ValidationError as e:
        print(f"Validation error in functions file: {e}")
        sys.exit(1)
    
    #2. load and validate prompts
    raw_prompts = load_json_file(args.input)
    try:
        prompts = [PromptInput(**p) for p in raw_prompts]
        print(f"Successfully loaded {len(prompts)} prompts.")
    except ValidationError as e:
        print(f"Validation error in prompts file: {e}")
        sys.exit(1)
    
    # 3. Initialize the LLM
    print("Initializing the LLM model...")
    model = Small_LLM_Model()

    # Get the path to the vocabulary file to see how tokens are mapped
    vocab_path = model.get_path_to_vocab_file()
    print(f"Vocabulary loaded from: {vocab_path}")

    # 4. Process each prompt
    results = []
    for p in prompts:
        print(f"\nProcessing prompt: {p.prompt}")

        # We need to start with a string that helps the model understand
        # it should output JSON
        starting_text = f"Question: {p.prompt}\nFunction Call: {{"

        # Turn the text into a list of numbers (IDs)
        # Note: input_ids from SDK encode() is a tensor,
        # but get_logits_from_input_ids() expects a list of ints. 
        input_ids_tensor = model.encode(starting_text)
        input_ids = input_ids_tensor[0].tolist()

        # This is where we will implement the constrained decoding loop
        # to generate the rest of the JSON string. 

        # TODO: Implement the token-by-token generation logic here
