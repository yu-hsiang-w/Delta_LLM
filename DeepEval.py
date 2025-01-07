from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import deepeval
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.1-8B"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp2/yhwang/cache")
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/tmp2/yhwang/cache").to("cuda")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load SQuAD dataset
print("Loading SQuAD dataset...")
data = load_dataset("squad", split="validation", cache_dir="/tmp2/yhwang/cache")

# Prepare the data for evaluation
def prepare_data(example):
    question = example["question"]
    answer = example["answers"]["text"]
    answer = answer[0] if isinstance(answer, list) and len(answer) > 0 else ""
    return {"prompt": f"Question: {question}\nAnswer:", "ground_truth": answer}

prepared_data = data.map(prepare_data)

# Define an evaluation function
def evaluate_fn(prompts):
    batch_size = 64  # Adjust to fit your GPU
    all_answers = []
    for i in range(0, len(prompts), batch_size):
        print(i)
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = model.generate(**inputs, max_length=100)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers = [pred.split("Answer:", 1)[-1].strip() for pred in predictions]
        all_answers.extend(answers)
    return all_answers

# Generate predictions
print("Generating predictions...")
prompts = [entry["prompt"] for entry in prepared_data]
predictions = evaluate_fn(prompts)

# Create test cases
print("Creating test cases...")
test_cases = [
    LLMTestCase(input=prompts[i], actual_output=predictions[i], expected_output=prepared_data[i]["ground_truth"])
    for i in range(len(prompts))
]

# Create evaluation dataset
dataset = EvaluationDataset(test_cases=test_cases)

# Define the metric
answer_relevancy_metric = AnswerRelevancyMetric(model="gpt-4o-mini")

# Evaluate the dataset
print("Evaluating the dataset...")
results = deepeval.evaluate(dataset, [answer_relevancy_metric])

# Print results
print("Evaluation Results (Answer Relevancy):")
print(results)
