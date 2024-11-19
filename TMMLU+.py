from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm

model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp2/yhwang/cache")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/tmp2/yhwang/cache").to("cuda")
model.eval()

task_list = [
             'engineering_math', 'dentistry', 'traditional_chinese_medicine_clinical_medicine', 'clinical_psychology', 'technical', 'culinary_skills', 'mechanical', 'logic_reasoning', 'real_estate',
             'general_principles_of_law', 'finance_banking', 'anti_money_laundering', 'ttqav2', 'marketing_management', 'business_management', 'organic_chemistry', 'advance_chemistry',
             'physics', 'secondary_physics', 'human_behavior', 'national_protection', 'jce_humanities', 'politic_science', 'agriculture', 'official_document_management',
             'financial_analysis', 'pharmacy', 'educational_psychology', 'statistics_and_machine_learning', 'management_accounting', 'introduction_to_law', 'computer_science', 'veterinary_pathology',
             'accounting', 'fire_science', 'optometry', 'insurance_studies', 'pharmacology', 'taxation', 'trust_practice', 'geography_of_taiwan', 'physical_education', 'auditing', 'administrative_law',
             'education_(profession_level)', 'economics', 'veterinary_pharmacology', 'nautical_science', 'occupational_therapy_for_psychological_disorders',
             'basic_medical_science', 'macroeconomics', 'trade', 'chinese_language_and_literature', 'tve_design', 'junior_science_exam', 'junior_math_exam', 'junior_chinese_exam',
             'junior_social_studies', 'tve_mathematics', 'tve_chinese_language', 'tve_natural_sciences', 'junior_chemistry', 'music', 'education', 'three_principles_of_people',
             'taiwanese_hokkien'
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
    dataset = load_dataset('ikala/tmmluplus', task, cache_dir="/tmp2/yhwang/cache")
    test_set = dataset['test']
    
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
file_path = "output.txt"

# Write the dictionary to a text file
with open(file_path, 'w') as file:
    for key, value in results.items():
        file.write(f"{key}: {value}\n")

#Aggregate results
overall_correct = sum([correct * len(test_set) for task, correct in results.items()])
overall_total = sum([len(load_dataset('ikala/tmmluplus', task, cache_dir="/tmp2/yhwang/cache")['test']) for task in task_list])
overall_accuracy = sum([results[task] * len(load_dataset('ikala/tmmluplus', task, cache_dir="/tmp2/yhwang/cache")['test']) for task in task_list]) / overall_total if overall_total > 0 else 0
print(f"Overall Accuracy across all tasks: {overall_accuracy:.2%}")
