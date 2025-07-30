#!/usr/bin/env python3
"""
Script to upload Lean Six Sigma Datasets (QnA or NER) to Hugging Face Datasets Hub
"""

import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import pandas as pd
from dotenv import load_dotenv

def load_and_validate_dataset(file_path, dataset_type):
    """Load and validate the dataset (QnA or NER)"""
    print(f"Loading {dataset_type} dataset from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Validate required fields based on dataset type
    if dataset_type == 'QnA':
        required_fields = ['id', 'instruction', 'input', 'output', 'type_of_question', 'sub_domain']
    elif dataset_type == 'NER':
        required_fields = ['id', 'instruction', 'input', 'output', 'type_of_question', 'sub_domain']
        # Note: NER has same structure but different output format (entities vs text)
    
    for i, sample in enumerate(data):
        missing_fields = [field for field in required_fields if field not in sample]
        if missing_fields:
            print(f"Warning: Sample {i+1} missing fields: {missing_fields}")
    
    return data

def create_dataset_card(dataset_type):
    """Create a comprehensive dataset card for the Hugging Face Hub"""
    
    if dataset_type == 'QnA':
        return create_qna_dataset_card()
    elif dataset_type == 'NER':
        return create_ner_dataset_card()

def create_qna_dataset_card():
    """Create dataset card for QnA dataset"""
    return """---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- lean-six-sigma
- business-consulting
- process-improvement
- supply-chain
- manufacturing
- quality-management
- DMAIC
- instruction-following
size_categories:
- n<1K
---

# Lean Six Sigma QnA Dataset

## Dataset Description

This dataset contains 102 high-quality question-answer pairs focused on Lean Six Sigma methodologies, business process improvement, and supply chain optimization. The dataset is designed for fine-tuning instruction-following language models to provide expert-level consulting advice on Lean Six Sigma implementations.

## Dataset Structure

### Data Fields

- **id**: Unique identifier for each sample (1-102)
- **instruction**: The question or problem statement requiring Lean Six Sigma expertise
- **input**: Additional context or data provided with the question (may be empty)
- **output**: Detailed, expert-level response following Lean Six Sigma methodologies
- **type_of_question**: Category of question (`consulting`, `methodology`)
- **sub_domain**: Specific area within Lean Six Sigma (e.g., `cycle_time_reduction`, `supply_chain_visibility`, `warehouse_productivity`)

### Data Splits

This dataset contains 102 samples provided as a single training split. Users can create their own validation/test splits based on their specific needs:
- **Full training**: Use all 102 samples for maximum data utilization
- **Custom splits**: Split by sub-domain, question type, or random sampling
- **Cross-validation**: Implement k-fold validation for robust evaluation

## Sub-domains Covered

The dataset covers diverse Lean Six Sigma applications including:

### Supply Chain & Logistics
- Material handling optimization
- Supply chain visibility enhancement
- Production planning improvement
- Cold chain logistics management
- Cross-docking operations
- Reverse logistics optimization
- Last-mile delivery enhancement
- Route optimization
- Order fulfillment efficiency

### Quality & Process Improvement
- Cycle time reduction
- Flow optimization
- Supplier quality management
- Demand forecasting accuracy
- Procurement efficiency
- Distribution optimization
- Warehouse productivity
- Inventory management
- Freight optimization

### Specialized Areas
- Sustainable supply chain practices
- Trade compliance optimization
- Supply chain resilience building

## Usage Examples

### Loading the Dataset

```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("your-username/lean-six-sigma-qna")['train']

# Option 1: Random split
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Option 2: Split by sub-domain (ensure domain coverage in validation)
unique_domains = set(dataset['sub_domain'])
val_domains = ['supply_chain_visibility', 'warehouse_productivity']  # Choose domains for validation
val_data = dataset.filter(lambda x: x['sub_domain'] in val_domains)
train_data = dataset.filter(lambda x: x['sub_domain'] not in val_domains)

# Option 3: Use all data for training (recommended for small datasets)
train_data = dataset
```

### Example Training Code (Alpaca Format)

```python
def format_alpaca_prompt(sample):
    instruction = sample["instruction"]
    input_text = sample["input"]
    
    if input_text.strip():
        return f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{sample["output"]}'''
    else:
        return f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{sample["output"]}'''

# Apply formatting
formatted_dataset = dataset.map(lambda x: {"text": format_alpaca_prompt(x)})
```

## Dataset Creation

This dataset was carefully curated to provide comprehensive coverage of Lean Six Sigma methodologies with:

- **Expert-level responses**: All outputs follow proper DMAIC (Define, Measure, Analyze, Improve, Control) methodology
- **Real-world scenarios**: Questions based on actual business challenges and case studies
- **Practical guidance**: Responses include specific tools, techniques, and implementation strategies
- **Supply chain focus**: Enhanced coverage of logistics and supply chain optimization scenarios

## Intended Use

This dataset is intended for:

1. **Fine-tuning instruction-following models** (3B-8B parameters) for Lean Six Sigma consulting
2. **Training business process improvement assistants**
3. **Developing domain-specific chatbots** for manufacturing and supply chain optimization
4. **Educational applications** in business process improvement training

## Model Performance

Recommended models for fine-tuning:
- **Llama 3.2 3B**: Optimal for 6GB VRAM GPUs (2-3 hour training)
- **Mistral 7B**: Excellent instruction following (1.5-2 hours on T4)
- **Qwen 2.5 7B**: Strong reasoning capabilities (1-1.5 hours on T4)

## Limitations

- Limited to 102 samples (suitable for parameter-efficient fine-tuning)
- Focused primarily on supply chain and manufacturing domains
- English language only
- Requires domain expertise to evaluate response quality

## Citation

If you use this dataset in your research, please cite:

```
@dataset{lean_six_sigma_qna_2025,
  title={Lean Six Sigma QnA Dataset},
  author={Clarence Wong},
  year={2025},
  url={https://huggingface.co/datasets/cw18/lean-six-sigma-qna}
}
```

## License

This dataset is released under the MIT License, allowing for both commercial and non-commercial use.
"""

def create_ner_dataset_card():
    """Create dataset card for NER dataset"""
    return """---
license: mit
task_categories:
- token-classification
- named-entity-recognition
- text-generation
language:
- en
tags:
- lean-six-sigma
- business-consulting
- process-improvement
- supply-chain
- manufacturing
- quality-management
- DMAIC
- NER
- entity-extraction
size_categories:
- n<1K
---

# Lean Six Sigma NER Dataset

## Dataset Description

This dataset contains 102 high-quality Named Entity Recognition (NER) samples focused on Lean Six Sigma methodologies, business process improvement, and supply chain optimization. Each sample identifies and categorizes key entities, tools, and methodologies within DMAIC (Define, Measure, Analyze, Improve, Control) framework responses.

## Dataset Structure

### Data Fields

- **id**: Unique identifier for each sample (1-102)
- **instruction**: The question or problem statement requiring Lean Six Sigma expertise
- **input**: Additional context or data provided with the question (may be empty)
- **output**: Dictionary mapping identified entities to their DMAIC phase categories
- **type_of_question**: Category of question (`consulting`, `methodology`)
- **sub_domain**: Specific area within Lean Six Sigma (e.g., `cycle_time_reduction`, `supply_chain_visibility`, `warehouse_productivity`)

### Entity Categories

The NER output categorizes Lean Six Sigma entities into DMAIC phases:
- **define**: Project definition, scope, and stakeholder identification activities
- **measure**: Data collection, baseline metrics, and measurement system activities
- **analyze**: Root cause analysis, statistical analysis, and process evaluation activities
- **improve**: Solution implementation, process optimization, and enhancement activities
- **control**: Monitoring, sustainment, and continuous improvement activities

### Data Splits

This dataset contains 102 samples provided as a single training split. Users can create their own validation/test splits based on their specific needs:
- **Full training**: Use all 102 samples for maximum data utilization
- **Custom splits**: Split by sub-domain, question type, or random sampling
- **Cross-validation**: Implement k-fold validation for robust evaluation

## Sub-domains Covered

The dataset covers diverse Lean Six Sigma applications including:

### Supply Chain & Logistics
- Material handling optimization
- Supply chain visibility enhancement
- Production planning improvement
- Cold chain logistics management
- Cross-docking operations
- Reverse logistics optimization
- Last-mile delivery enhancement
- Route optimization
- Order fulfillment efficiency

### Quality & Process Improvement
- Cycle time reduction
- Flow optimization
- Supplier quality management
- Demand forecasting accuracy
- Procurement efficiency
- Distribution optimization
- Warehouse productivity
- Inventory management
- Freight optimization

### Specialized Areas
- Sustainable supply chain practices
- Trade compliance optimization
- Supply chain resilience building

## Usage Examples

### Loading the Dataset

```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("your-username/lean-six-sigma-ner")['train']

# Option 1: Use all data for training (recommended for small datasets)
train_data = dataset

# Option 2: Random split
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Option 3: Split by sub-domain for domain-aware validation
unique_domains = set(dataset['sub_domain'])
val_domains = ['supply_chain_visibility', 'warehouse_productivity']
val_data = dataset.filter(lambda x: x['sub_domain'] in val_domains)
train_data = dataset.filter(lambda x: x['sub_domain'] not in val_domains)
```

### Example Entity Extraction

```python
# Example sample structure
sample = dataset[0]
print(f"Instruction: {sample['instruction']}")
print(f"Entities by DMAIC phase:")
for entity, phases in sample['output'].items():
    print(f"  {entity}: {phases}")

# Extract entities for a specific DMAIC phase
define_entities = [entity for entity, phases in sample['output'].items() if 'define' in phases]
print(f"Define phase entities: {define_entities}")
```

### Training for Entity Recognition

```python
def format_ner_prompt(sample):
    instruction = sample["instruction"]
    input_text = sample["input"]
    entities = sample["output"]
    
    # Create entity list for training
    entity_text = "\\n".join([f"- {entity}: {', '.join(phases)}" for entity, phases in entities.items()])
    
    if input_text.strip():
        return f'''Extract Lean Six Sigma entities and categorize them by DMAIC phase.

### Context:
{input_text}

### Question:
{instruction}

### Entities:
{entity_text}'''
    else:
        return f'''Extract Lean Six Sigma entities and categorize them by DMAIC phase.

### Question:
{instruction}

### Entities:
{entity_text}'''

# Apply formatting
formatted_dataset = dataset.map(lambda x: {"text": format_ner_prompt(x)})
```

## Dataset Creation

This NER dataset was carefully aligned with the corresponding QnA dataset to provide:

- **DMAIC-aligned entities**: All entities are categorized according to the DMAIC methodology
- **Real-world terminology**: Entities extracted from actual business scenarios and case studies
- **Comprehensive coverage**: Spans all major Lean Six Sigma tools and techniques
- **Supply chain focus**: Enhanced coverage of logistics and supply chain optimization entities

## Intended Use

This dataset is intended for:

1. **Training NER models** for Lean Six Sigma entity extraction
2. **Fine-tuning language models** for domain-specific entity recognition
3. **Developing knowledge extraction systems** for business process improvement
4. **Educational applications** in Lean Six Sigma methodology training
5. **Paired training** with the corresponding QnA dataset for comprehensive understanding

## Model Performance

Recommended approaches:
- **Sequence labeling**: Use with BERT-based models for token classification
- **Generative NER**: Fine-tune instruction-following models for entity extraction
- **Multi-task learning**: Combine with QnA dataset for comprehensive Lean Six Sigma understanding

## Limitations

- Limited to 102 samples (suitable for few-shot learning and fine-tuning)
- Focused primarily on supply chain and manufacturing domains
- English language only
- Requires domain expertise to evaluate entity categorization quality
- Entity categories limited to DMAIC framework phases

## Citation

If you use this dataset in your research, please cite:

```
@dataset{lean_six_sigma_ner_2025,
  title={Lean Six Sigma NER Dataset},
  author={Clarence Wong},
  year={2025},
  url={https://huggingface.co/datasets/cw18/lean-six-sigma-ner}
}
```

## License

This dataset is released under the MIT License, allowing for both commercial and non-commercial use.
"""

def prepare_dataset_for_upload(data):
    """Convert JSON data to Hugging Face Dataset format"""
    print("Preparing dataset for upload...")
    
    # Convert to pandas DataFrame first for easier manipulation
    df = pd.DataFrame(data)
    
    # Create single train split - let users decide how to split
    print(f"Creating single train split with {len(data)} samples")
    
    # Create Dataset object
    train_dataset = Dataset.from_pandas(df)
    
    # Create DatasetDict with only train split
    dataset_dict = DatasetDict({
        'train': train_dataset
    })
    
    return dataset_dict

def upload_to_huggingface(dataset_dict, repo_name, dataset_type, private=False):
    """Upload dataset to Hugging Face Hub"""
    print(f"Uploading {dataset_type} dataset to: {repo_name}")
    
    # Create dataset card
    card_content = create_dataset_card(dataset_type)
    
    # Set task categories based on dataset type
    if dataset_type == 'QnA':
        task_categories = ["question-answering", "text-generation"]
        tags = ["lean-six-sigma", "business-consulting", "process-improvement", 
               "supply-chain", "manufacturing", "quality-management", "DMAIC", 
               "instruction-following"]
    elif dataset_type == 'NER':
        task_categories = ["token-classification", "named-entity-recognition", "text-generation"]
        tags = ["lean-six-sigma", "business-consulting", "process-improvement", 
               "supply-chain", "manufacturing", "quality-management", "DMAIC", 
               "NER", "entity-extraction"]
    
    try:
        # Push to hub (without card_data for compatibility)
        dataset_dict.push_to_hub(
            repo_id=repo_name,
            private=private
        )
        
        # Upload README separately to ensure it's properly formatted
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset"
        )
        
        print(f"✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_name}")
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        return False
    
    return True

def main():
    """Main execution function"""
    print("🚀 Lean Six Sigma Dataset Upload Script")
    print("=" * 50)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Dataset selection
    print("\n📊 Select Dataset Type:")
    print("1. QnA Dataset (Question-Answer pairs)")
    print("2. NER Dataset (Named Entity Recognition)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice == '1':
            dataset_type = 'QnA'
            dataset_path = r"c:\Users\clwong\OneDrive\Documents\Learning\AI_MasterBlackBelt\datasets\lss_consultant\sixSigma_QnA_caseStudy_sample.json"
            default_name = "lean-six-sigma-qna"
            break
        elif choice == '2':
            dataset_type = 'NER'
            dataset_path = r"c:\Users\clwong\OneDrive\Documents\Learning\AI_MasterBlackBelt\datasets\lss_ner\sixSigma_NER_caseStudy_sample.json"
            default_name = "lean-six-sigma-ner"
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    
    print(f"\n✅ Selected: {dataset_type} Dataset")
    print(f"📁 Source file: {dataset_path}")
    
    # Get user configuration
    print("\n📝 Configuration:")
    
    # Try to get username from environment variable
    default_username = os.getenv('HF_USERNAME') or os.getenv('HUGGINGFACE_USERNAME')
    
    if default_username:
        username = input(f"Enter your Hugging Face username (default: {default_username}): ").strip()
        if not username:
            username = default_username
        print(f"✅ Using username: {username}")
    else:
        username = input("Enter your Hugging Face username: ").strip()
        if not username:
            print("❌ Username is required!")
            return
    
    dataset_name = input(f"Enter dataset name (default: {default_name}): ").strip()
    if not dataset_name:
        dataset_name = default_name
    
    repo_name = f"{username}/{dataset_name}"
    
    private = input("Make dataset private? (y/N): ").strip().lower() == 'y'
    
    print(f"\n📊 Dataset will be uploaded to: {repo_name}")
    print(f"🔒 Private: {private}")
    
    # Authenticate with Hugging Face
    print("\n🔐 Authenticating with Hugging Face...")
    
    # Try to get token from environment variable first
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    try:
        if hf_token:
            print("✅ Found token in environment variables")
            login(token=hf_token)
            print("✅ Authentication successful!")
        else:
            print("ℹ️  No token found in .env file, using interactive login...")
            login()
            print("✅ Authentication successful!")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        if not hf_token:
            print("\n💡 Tip: Create a .env file with your token:")
            print("   HUGGINGFACE_TOKEN=your_write_token_here")
        print("   Or run 'huggingface-cli login' in your terminal first.")
        return
    
    # Load and validate dataset
    try:
        data = load_and_validate_dataset(dataset_path, dataset_type)
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Prepare dataset
    try:
        dataset_dict = prepare_dataset_for_upload(data)
        print("✅ Dataset prepared successfully!")
    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")
        return
    
    # Upload to Hugging Face
    print(f"\n📤 Uploading to Hugging Face Hub...")
    success = upload_to_huggingface(dataset_dict, repo_name, dataset_type, private)
    
    if success:
        print("\n🎉 Upload completed successfully!")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{repo_name}")
        print(f"📚 You can now use: load_dataset('{repo_name}')")
        
        # # Create example usage file based on dataset type
        # if dataset_type == 'QnA':
        #     example_code = create_qna_example_code(repo_name)
        # elif dataset_type == 'NER':
        #     example_code = create_ner_example_code(repo_name)
        
        # filename = f"example_usage_{dataset_type.lower()}.py"
        # with open(filename, "w", encoding='utf-8') as f:
        #     f.write(example_code)
        
        # print(f"📝 Created {filename} for reference")
    else:
        print("\n❌ Upload failed. Please check the error messages above.")

def create_qna_example_code(repo_name):
    """Create example usage code for QnA dataset"""
    return f'''# Example: Loading and using your QnA dataset

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load the dataset (single train split)
dataset = load_dataset("{repo_name}")['train']

print(f"Total samples: {{len(dataset)}}")

# Option 1: Use all data for training (recommended for small datasets)
train_data = dataset
print("Using all 102 samples for training")

# Option 2: Create random validation split
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
print(f"Random split - Train: {{len(train_data)}}, Val: {{len(val_data)}}")

# Option 3: Split by sub-domain for domain-aware validation
unique_domains = set(dataset['sub_domain'])
print(f"Available sub-domains: {{sorted(unique_domains)}}")

# Example: Reserve specific domains for validation
val_domains = ['supply_chain_visibility', 'warehouse_productivity']
val_data = dataset.filter(lambda x: x['sub_domain'] in val_domains)
train_data = dataset.filter(lambda x: x['sub_domain'] not in val_domains)
print(f"Domain split - Train: {{len(train_data)}}, Val: {{len(val_data)}}")

# Example sample
sample = dataset[0]
print("\\nExample sample:")
print(f"ID: {{sample['id']}}")
print(f"Sub-domain: {{sample['sub_domain']}}")
print(f"Question type: {{sample['type_of_question']}}")
print(f"Instruction: {{sample['instruction'][:100]}}...")

# Alpaca formatting function
def format_alpaca_prompt(sample):
    instruction = sample["instruction"]
    input_text = sample["input"]
    
    if input_text.strip():
        return f\"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{instruction}}

### Input:
{{input_text}}

### Response:
{{sample["output"]}}\"\"\"
    else:
        return f\"\"\"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{instruction}}

### Response:
{{sample["output"]}}\"\"\"

# Apply formatting for training
formatted_dataset = dataset.map(lambda x: {{"text": format_alpaca_prompt(x)}})
print(f"\\nFormatted for training: {{len(formatted_dataset)}} samples")
'''

def create_ner_example_code(repo_name):
    """Create example usage code for NER dataset"""
    return f'''# Example: Loading and using your NER dataset

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

# Load the dataset (single train split)
dataset = load_dataset("{repo_name}")['train']

print(f"Total samples: {{len(dataset)}}")

# Option 1: Use all data for training (recommended for small datasets)
train_data = dataset
print("Using all 102 samples for training")

# Option 2: Create random validation split
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
print(f"Random split - Train: {{len(train_data)}}, Val: {{len(val_data)}}")

# Option 3: Split by sub-domain for domain-aware validation
unique_domains = set(dataset['sub_domain'])
print(f"Available sub-domains: {{sorted(unique_domains)}}")

# Example: Reserve specific domains for validation
val_domains = ['supply_chain_visibility', 'warehouse_productivity']
val_data = dataset.filter(lambda x: x['sub_domain'] in val_domains)
train_data = dataset.filter(lambda x: x['sub_domain'] not in val_domains)
print(f"Domain split - Train: {{len(train_data)}}, Val: {{len(val_data)}}")

# Example sample
sample = dataset[0]
print("\\nExample sample:")
print(f"ID: {{sample['id']}}")
print(f"Sub-domain: {{sample['sub_domain']}}")
print(f"Question type: {{sample['type_of_question']}}")
print(f"Instruction: {{sample['instruction'][:100]}}...")
print("\\nEntities by DMAIC phase:")
for entity, phases in sample['output'].items():
    print(f"  {{entity}}: {{phases}}")

# Entity extraction functions
def extract_entities_by_phase(sample, target_phase):
    \"\"\"Extract entities for a specific DMAIC phase\"\"\"
    return [entity for entity, phases in sample['output'].items() if target_phase in phases]

def get_all_entities(sample):
    \"\"\"Get all entities from a sample\"\"\"
    return list(sample['output'].keys())

def get_all_phases(sample):
    \"\"\"Get all DMAIC phases mentioned in a sample\"\"\"
    phases = set()
    for entity_phases in sample['output'].values():
        phases.update(entity_phases)
    return sorted(list(phases))

# Example usage
define_entities = extract_entities_by_phase(sample, 'define')
print(f"\\nDefine phase entities: {{define_entities}}")

all_entities = get_all_entities(sample)
print(f"All entities in sample: {{all_entities[:5]}}...")  # Show first 5

all_phases = get_all_phases(sample)
print(f"All DMAIC phases: {{all_phases}}")

# NER formatting function for training
def format_ner_prompt(sample):
    instruction = sample["instruction"]
    input_text = sample["input"]
    entities = sample["output"]
    
    # Create entity list for training
    entity_text = "\\n".join([f"- {{entity}}: {{', '.join(phases)}}" for entity, phases in entities.items()])
    
    if input_text.strip():
        return f\"\"\"Extract Lean Six Sigma entities and categorize them by DMAIC phase.

### Context:
{{input_text}}

### Question:
{{instruction}}

### Entities:
{{entity_text}}\"\"\"
    else:
        return f\"\"\"Extract Lean Six Sigma entities and categorize them by DMAIC phase.

### Question:
{{instruction}}

### Entities:
{{entity_text}}\"\"\"

# Apply formatting for training
formatted_dataset = dataset.map(lambda x: {{"text": format_ner_prompt(x)}})
print(f"\\nFormatted for training: {{len(formatted_dataset)}} samples")

# Statistics
total_entities = sum(len(sample['output']) for sample in dataset)
print(f"\\nDataset statistics:")
print(f"Total entities across all samples: {{total_entities}}")
print(f"Average entities per sample: {{total_entities / len(dataset):.1f}}")

# Phase distribution
phase_counts = {{}}
for sample in dataset:
    for entity_phases in sample['output'].values():
        for phase in entity_phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

print(f"\\nEntity distribution by DMAIC phase:")
for phase, count in sorted(phase_counts.items()):
    print(f"  {{phase}}: {{count}} entities")
'''

if __name__ == "__main__":
    main()
