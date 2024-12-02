from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm

#model_name = "meta-llama/Llama-3.1-8B"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp2/yhwang/cache")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/tmp2/yhwang/cache").to("cuda")
model.eval()

task_list = [
                "computer_network",
                "operating_system",
                "computer_architecture",
                "college_programming",
                "college_physics",
                "college_chemistry",
                "advanced_mathematics",
                "probability_and_statistics",
                "discrete_mathematics",
                "electrical_engineer",
                "metrology_engineer",
                "high_school_mathematics",
                "high_school_physics",
                "high_school_chemistry",
                "high_school_biology",
                "middle_school_mathematics",
                "middle_school_biology",
                "middle_school_physics",
                "middle_school_chemistry",
                "veterinary_medicine",
                "college_economics",
                "business_administration",
                "marxism",
                "mao_zedong_thought",
                "education_science",
                "teacher_qualification",
                "high_school_politics",
                "high_school_geography",
                "middle_school_politics",
                "middle_school_geography",
                "modern_chinese_history",
                "ideological_and_moral_cultivation",
                "logic",
                "law",
                "chinese_language_and_literature",
                "art_studies",
                "professional_tour_guide",
                "legal_professional",
                "high_school_chinese",
                "high_school_history",
                "middle_school_history",
                "civil_servant",
                "sports_science",
                "plant_protection",
                "basic_medicine",
                "clinical_medicine",
                "urban_and_rural_planner",
                "accountant",
                "fire_engineer",
                "environmental_impact_assessment_engineer",
                "tax_accountant",
                "physician"
            ]

# Initialize a dictionary to store results
results = {}

# Function to extract the predicted answer from the model's output
def extract_answer(generated_text):
    # Possible answer choices
    choices = ['A', 'B', 'C', 'D']
    # Convert generated text to uppercase to standardize
    text = generated_text.upper()
    for choice in choices:
        if choice in text:
            return choice
    return None  # If no valid choice is found

# Iterate over each task
for task in task_list:
    print(f"Evaluating task: {task}")
    
    # Load dataset splits
    dataset = load_dataset("ceval/ceval-exam", task, cache_dir="/tmp2/yhwang/cache")
    test_set = dataset['val']

    correct = 0
    total = 0
    
    # Use tqdm for a progress bar
    for row in tqdm(test_set, desc=f"Processing {task}"):
        # Prepare the input prompt
        prompt = f"{row['question']}\n(A) {row['A']}\n(B) {row['B']}\n(C) {row['C']}\n(D) {row['D']}\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate output
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_tokens = output[0][inputs['input_ids'].shape[-1]:]  # Slice tokens after the prompt
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        #print(generated_text)
        
        # Extract the predicted answer
        pred = extract_answer(generated_text)
        
        # Compare with the true answer
        true_answer = row['answer'].strip().upper()
        if pred == true_answer:
            correct += 1
        total += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    results[task] = accuracy
    print(f"Task: {task}, Accuracy: {accuracy:.2%}\n")

for key in results:
    print(f"{key}: {results[key]}")

# Specify the file path
file_path = "output_llama_8b_instruct_ceval.txt"

# Write the dictionary to a text file
with open(file_path, 'w') as file:
    for key, value in results.items():
        file.write(f"{key}: {value}\n")


#Aggregate results
overall_correct = sum([correct * len(test_set) for task, correct in results.items()])
overall_total = sum([len(load_dataset("ceval/ceval-exam", task, cache_dir="/tmp2/yhwang/cache")) for task in task_list])
overall_accuracy = sum([results[task] * len(load_dataset("ceval/ceval-exam", task, cache_dir="/tmp2/yhwang/cache")) for task in task_list]) / overall_total if overall_total > 0 else 0
print(f"Overall Accuracy across all tasks: {overall_accuracy:.2%}")
