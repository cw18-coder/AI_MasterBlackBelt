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

def load_and_validate_cot_samples(directory):
    """Collate and validate CoT samples from all batch files in the directory."""
    import re
    pattern = re.compile(r"lss_cot_batch(\d+)\.json")
    all_samples = []
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    print(f"Found {len(files)} CoT batch files in {directory}")
    for fname in sorted(files):
        fpath = os.path.join(directory, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                all_samples.extend(data)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
    print(f"Total CoT samples loaded: {len(all_samples)}")
    # Validate required fields
    required_fields = ['id', 'domain', 'sub_domain', 'instruction', 'input', 'output']
    for i, sample in enumerate(all_samples):
        missing_fields = [field for field in required_fields if field not in sample]
        if missing_fields:
            print(f"Warning: CoT sample {i+1} missing fields: {missing_fields}")
    return all_samples

def create_dataset_card(dataset_type):
    """Create a comprehensive dataset card for the Hugging Face Hub"""
    
    if dataset_type == 'QnA':
        return create_qna_dataset_card()
    elif dataset_type == 'NER':
        return create_ner_dataset_card()
    elif dataset_type == 'CoT':
        return create_cot_dataset_card()

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
- data-center
- healthcare
- ecommerce
- energy-utilities
- DMAIC
- instruction-following
size_categories:
- n<1K
---

# Lean Six Sigma QnA Dataset

## Dataset Description

This dataset contains 360 high-quality question-answer pairs focused on Lean Six Sigma methodologies, business process improvement, and operational optimization across multiple industries. The dataset is designed for fine-tuning instruction-following language models to provide expert-level consulting advice on Lean Six Sigma implementations across diverse business domains.

## Dataset Structure

### Data Fields

- **id**: Unique identifier for each sample (1-360)
- **instruction**: The question or problem statement requiring Lean Six Sigma expertise
- **input**: Additional context or data provided with the question (may be empty)
- **output**: Detailed, expert-level response following Lean Six Sigma methodologies
- **type_of_question**: Category of question (`consulting`, `methodology`)
- **sub_domain**: Specific area within Lean Six Sigma across various industries

### Data Splits

This dataset contains 360 samples provided as a single training split. Users can create their own validation/test splits based on their specific needs:
- **Full training**: Use all 360 samples for maximum data utilization
- **Custom splits**: Split by sub-domain, question type, or random sampling
- **Cross-validation**: Implement k-fold validation for robust evaluation
- **Domain-aware splits**: Reserve specific industries/domains for validation

## Industry Coverage

The dataset covers comprehensive Lean Six Sigma applications across six major domains:

### Healthcare Operations (60 samples)
- Patient flow optimization
- Medical process improvement
- Quality metrics enhancement
- Care delivery efficiency
- Clinical workflow optimization
- Healthcare resource management

### E-commerce Operations (60 samples)
- Customer experience optimization
- Order fulfillment enhancement
- Digital platform performance
- Conversion rate improvement
- Online retail process optimization
- Digital customer journey enhancement

### Manufacturing Operations (60 samples)
- Production line optimization
- Quality control enhancement
- Efficiency improvement
- Waste reduction strategies
- Manufacturing process improvement
- Industrial automation optimization

### Energy & Utilities (60 samples)
- Grid reliability optimization
- Energy efficiency enhancement
- Resource management improvement
- Infrastructure optimization
- Utility operations improvement
- Renewable energy integration

### Data Center Operations (60 samples)
- Infrastructure performance optimization
- Cloud migration and hybrid operations
- Container platform and database optimization
- AI/ML workload management
- Edge computing deployment
- Network performance enhancement
- Security and compliance optimization
- Automation and change management
- Environmental and power systems optimization
- Virtualization and storage efficiency

### Supply Chain & Logistics (60 samples)
- Material handling optimization
- Supply chain visibility enhancement
- Production planning improvement
- Reverse logistics optimization
- Last-mile delivery enhancement
- Procurement and supplier management
- Inventory and warehouse optimization
- Demand planning and forecasting
- Quality and risk management
- Omnichannel fulfillment
- Trade compliance and finance optimization

## Usage Examples

### Loading the Dataset

from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("your-username/lean-six-sigma-qna")['train']

# Option 1: Random split
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Option 2: Split by industry domain (ensure domain coverage in validation)
healthcare_samples = dataset.filter(lambda x: x['sub_domain'] == 'healthcare')
ecommerce_samples = dataset.filter(lambda x: x['sub_domain'] == 'ecommerce')
# Reserve one domain for validation
val_data = healthcare_samples
train_data = dataset.filter(lambda x: x['sub_domain'] != 'healthcare')

# Option 3: Use all data for training (recommended for comprehensive coverage)
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
- **Real-world scenarios**: Questions based on actual business challenges and case studies across multiple industries
- **Practical guidance**: Responses include specific tools, techniques, and implementation strategies
- **Multi-industry focus**: Comprehensive coverage across healthcare, e-commerce, manufacturing, energy, data centers, and supply chain
- **Modern techniques**: Integration of AI/ML, automation, digital transformation, and advanced analytics
- **Strategic balance**: Equal representation across major business domains (60 samples per domain)

## Intended Use

This dataset is intended for:

1. **Fine-tuning instruction-following models** (3B-8B parameters) for comprehensive Lean Six Sigma consulting
2. **Training multi-industry business process improvement assistants**
3. **Developing domain-specific chatbots** for operational optimization across industries
4. **Educational applications** in business process improvement training
5. **Cross-industry knowledge transfer** and best practice identification

## Model Performance

Recommended models for fine-tuning:
- **Llama 3.2 3B**: Optimal for 6GB VRAM GPUs (3-4 hour training)
- **Mistral 7B**: Excellent instruction following (2-3 hours on T4)
- **Qwen 2.5 7B**: Strong reasoning capabilities (1.5-2 hours on T4)
- **Gemma 7B**: Google's instruction-tuned model (2.5-3 hours on T4)

## Limitations

- Limited to 360 samples (optimized for parameter-efficient fine-tuning)
- English language only
- Requires domain expertise to evaluate response quality across different industries
- Focus on operational and process improvement domains

## Citation

If you use this dataset in your research, please cite:

```
@dataset{lean_six_sigma_qna_2025,
  title={Lean Six Sigma QnA Dataset},
  author={Clarence Wong},
  year={2025},
  url={https://huggingface.co/datasets/cw18/lean-six-sigma-qna},
  samples={360},
  domains={healthcare, ecommerce, manufacturing, energy_utilities, data_center_operations, supply_chain_logistics}
}
```

## License

This dataset is released under the MIT License, allowing for both commercial and non-commercial use.
"""

def create_cot_dataset_card():
    """Create dataset card for CoT dataset."""
    return """---
license: mit
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- lean-six-sigma
- chain-of-thought-reasoning
- business-consulting
- process-improvement
- multi-domain
- DMAIC
- advanced-reasoning
size_categories:
- n<10K
---

# Lean Six Sigma Chain-of-Thought (CoT) Reasoning Dataset

## Dataset Description

This dataset contains high-quality Chain-of-Thought (CoT) reasoning samples for Lean Six Sigma and statistical problem-solving. Each sample includes step-by-step expert reasoning, industry context, and method selection, covering DMAIC, hypothesis testing, FAQ, data reasoning, and mixed sample types. The dataset is designed for fine-tuning language models to perform advanced Chain-of-Thought reasoning and decision support in business process improvement.

## Dataset Structure

### Data Fields
- **id**: Unique identifier for each sample
- **batch**: Batch number
- **sample_type**: Type of sample (e.g., DMAIC, FAQ, hypothesis, data reasoning, mixed)
- **domain**: Industry domain
- **sub_domain**: Sub-domain within the industry
- **instruction**: Task or question
- **input**: Scenario or data context
- **output**: Chain-of-Thought reasoning and answer

### Data Splits
All samples are provided as a single training split. Users may create custom splits as needed.

## Industry Coverage

### Target Distribution

| Industry Category                   | Target % |
|-------------------------------------|----------|
| Manufacturing Industries            | 20.0%    |
| Transportation & Logistics          | 20.0%    |
| Technology & Data Center Operations | 20.0%    |
| Financial & Professional Services   | ~7.7%    |
| Healthcare & Life Sciences          | ~5.7%    |
| Energy & Utilities                  | ~5.7%    |
| Public Sector & Non-Profit          | ~5.7%    |
| Telecommunications & Media          | ~3.8%    |
| Retail & E-commerce                 | ~3.8%    |
| Hospitality & Services              | ~3.8%    |
| Construction & Infrastructure       | ~1.9%    |
| Aerospace & Defense                 | ~1.9%    |

The dataset is carefully balanced to maintain these target percentages, ensuring broad and representative coverage across all major business domains for Lean Six Sigma applications.

## Usage Examples

### Loading the Dataset

from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("your-username/lean-six-sigma-cot")['train']

# Option 1: Random split
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Option 2: Split by domain (ensure domain coverage in validation)
healthcare_samples = dataset.filter(lambda x: x['domain'] == 'healthcare')
val_data = healthcare_samples
train_data = dataset.filter(lambda x: x['domain'] != 'healthcare')

# Option 3: Use all data for training (recommended for comprehensive coverage)
train_data = dataset

### Example Training Code (Alpaca Format)

def format_cot_prompt(sample):
    instruction = sample["instruction"]
    input_text = sample["input"]
    reasoning = sample["output"]
    if input_text.strip():
        return f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a step-by-step Chain-of-Thought reasoning response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Chain-of-Thought Reasoning:\n{reasoning}'''
    else:
        return f'''Below is an instruction that describes a task. Write a step-by-step Chain-of-Thought reasoning response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Chain-of-Thought Reasoning:\n{reasoning}'''

# Apply formatting
formatted_dataset = dataset.map(lambda x: {"text": format_cot_prompt(x)})

## Dataset Creation

This dataset was carefully curated to provide comprehensive coverage of Lean Six Sigma methodologies and advanced Chain-of-Thought reasoning with:

- **Expert-level step-by-step reasoning**: All outputs follow proper DMAIC (Define, Measure, Analyze, Improve, Control) methodology and statistical best practices
- **Real-world scenarios**: Samples based on actual business challenges and case studies across multiple industries
- **Practical guidance**: Responses include specific tools, techniques, and implementation strategies with explicit reasoning
- **Multi-industry focus**: Comprehensive coverage across all major business domains
- **Modern techniques**: Integration of AI/ML, automation, digital transformation, and advanced analytics
- **Strategic balance**: Targeted distribution across all listed domains

## Intended Use

This dataset is intended for:

1. **Fine-tuning instruction-following models** (3B-8B parameters) for advanced Chain-of-Thought reasoning in Lean Six Sigma consulting
2. **Training multi-industry business process improvement assistants**
3. **Developing domain-specific chatbots** for operational optimization and decision support
4. **Educational applications** in business process improvement and statistical analysis
5. **Cross-industry knowledge transfer** and best practice identification

## Model Performance

Recommended models for fine-tuning:
- **Llama 3.2 3B**: Optimal for 6GB VRAM GPUs (3-4 hour training)
- **Mistral 7B**: Excellent instruction following and reasoning (2-3 hours on T4)
- **Qwen 2.5 7B**: Strong Chain-of-Thought capabilities (1.5-2 hours on T4)
- **Gemma 7B**: Google's instruction-tuned model (2.5-3 hours on T4)

## Limitations

- Limited to 360 samples (optimized for parameter-efficient fine-tuning)
- English language only
- Requires domain expertise to evaluate reasoning quality across different industries
- Focus on operational and process improvement domains

## Citation

If you use this dataset in your research, please cite:

@dataset{lean_six_sigma_cot_2025,
  title={Lean Six Sigma Chain-of-Thought Reasoning Dataset},
  author={Clarence Wong},
  year={2025},
  url={https://huggingface.co/datasets/cw18/lean-six-sigma-cot},
  samples={360},
  domains={manufacturing, transportation_logistics, technology_data_center, financial_professional_services, healthcare_life_sciences, energy_utilities, public_sector_non_profit, telecommunications_media, retail_ecommerce, hospitality_services, construction_infrastructure, aerospace_defense}
}

## License

This dataset is released under the MIT License, allowing for both commercial and non-commercial use.
"""

def create_ner_dataset_card():
    """Create dataset card for NER dataset"""
    return """---
license: mit
task_categories:
- token-classification
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
- data-center
- healthcare
- ecommerce
- energy-utilities
- DMAIC
- NER
- entity-extraction
- named-entity-recognition
size_categories:
- n<1K
---

# Lean Six Sigma NER Dataset

## Dataset Description

This dataset contains 360 high-quality Named Entity Recognition (NER) samples focused on Lean Six Sigma methodologies, business process improvement, and operational optimization across multiple industries. Each sample identifies and categorizes key entities, tools, and methodologies within DMAIC (Define, Measure, Analyze, Improve, Control) framework responses across diverse business domains.

## Dataset Structure

### Data Fields

- **id**: Unique identifier for each sample (1-360)
- **instruction**: The question or problem statement requiring Lean Six Sigma expertise
- **input**: Additional context or data provided with the question (may be empty)
- **output**: Dictionary mapping identified entities to their DMAIC phase categories
- **type_of_question**: Category of question (`consulting`, `methodology`)
- **sub_domain**: Specific area within Lean Six Sigma across various industries

### Entity Categories

The NER output categorizes Lean Six Sigma entities into DMAIC phases:
- **define**: Project definition, scope, stakeholder identification, and goal-setting activities
- **measure**: Data collection, baseline metrics, measurement system analysis activities
- **analyze**: Root cause analysis, statistical analysis, process evaluation activities
- **improve**: Solution implementation, process optimization, enhancement activities
- **control**: Monitoring, sustainment, governance, and continuous improvement activities

### Data Splits

This dataset contains 360 samples provided as a single training split. Users can create their own validation/test splits based on their specific needs:
- **Full training**: Use all 360 samples for maximum data utilization
- **Custom splits**: Split by sub-domain, question type, or random sampling
- **Cross-validation**: Implement k-fold validation for robust evaluation
- **Domain-aware splits**: Reserve specific industries/domains for validation

## Industry Coverage

The dataset covers comprehensive Lean Six Sigma entity extraction across six major domains:

### Healthcare Operations (60 samples)
- Patient flow optimization entities
- Medical process improvement tools
- Quality metrics and healthcare KPIs
- Care delivery efficiency techniques
- Clinical workflow optimization methods
- Healthcare resource management tools

### E-commerce Operations (60 samples)
- Customer experience optimization entities
- Order fulfillment enhancement tools
- Digital platform performance metrics
- Conversion rate improvement techniques
- Online retail process optimization methods
- Digital customer journey analysis tools

### Manufacturing Operations (60 samples)
- Production line optimization entities
- Quality control enhancement tools
- Efficiency improvement techniques
- Waste reduction methodologies
- Manufacturing process improvement tools
- Industrial automation optimization methods

### Energy & Utilities (60 samples)
- Grid reliability optimization entities
- Energy efficiency enhancement tools
- Resource management improvement techniques
- Infrastructure optimization methods
- Utility operations improvement tools
- Renewable energy integration techniques

### Data Center Operations (60 samples)
- Infrastructure performance optimization entities
- Cloud migration and hybrid operations tools
- Container platform and database optimization techniques
- AI/ML workload management methods
- Edge computing deployment tools
- Network performance enhancement techniques
- Security and compliance optimization tools
- Automation and change management methods
- Environmental and power systems optimization entities
- Virtualization and storage efficiency tools

### Supply Chain & Logistics (60 samples)
- Material handling optimization entities
- Supply chain visibility enhancement tools
- Production planning improvement techniques
- Reverse logistics optimization methods
- Last-mile delivery enhancement tools
- Procurement and supplier management entities
- Inventory and warehouse optimization techniques
- Demand planning and forecasting tools
- Quality and risk management methods
- Omnichannel fulfillment optimization entities
- Trade compliance and finance optimization tools

## Usage Examples

### Loading the Dataset

```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("your-username/lean-six-sigma-ner")['train']

# Option 1: Use all data for training (recommended for comprehensive coverage)
train_data = dataset

# Option 2: Random split
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Option 3: Split by industry domain for domain-aware validation
healthcare_samples = dataset.filter(lambda x: x['sub_domain'] == 'healthcare')
val_data = healthcare_samples
train_data = dataset.filter(lambda x: x['sub_domain'] != 'healthcare')
```

### Example Entity Extraction

```python
# Example sample structure
sample = dataset[0]
print(f"Instruction: {sample['instruction']}")
print(f"Industry Domain: {sample['sub_domain']}")
print(f"Entities by DMAIC phase:")
for entity, phases in sample['output'].items():
    print(f"  {entity}: {phases}")

# Extract entities for a specific DMAIC phase
define_entities = [entity for entity, phases in sample['output'].items() if 'define' in phases]
print(f"Define phase entities: {define_entities}")

# Extract entities across all samples for a specific industry
healthcare_entities = set()
for sample in dataset.filter(lambda x: x['sub_domain'] == 'healthcare'):
    healthcare_entities.update(sample['output'].keys())
print(f"Healthcare domain entities: {list(healthcare_entities)[:10]}...")  # Show first 10
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

### Industry Context:
{sample["sub_domain"]}

### Scenario:
{input_text}

### Question:
{instruction}

### Entities:
{entity_text}'''
    else:
        return f'''Extract Lean Six Sigma entities and categorize them by DMAIC phase.

### Industry Context:
{sample["sub_domain"]}

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
- **Real-world terminology**: Entities extracted from actual business scenarios across multiple industries
- **Comprehensive coverage**: Spans all major Lean Six Sigma tools and techniques across six domains
- **Multi-industry focus**: Equal representation across healthcare, e-commerce, manufacturing, energy, data centers, and supply chain
- **Modern techniques**: Advanced entities including AI/ML, automation, digital transformation, and analytics tools
- **Perfect alignment**: Complete correspondence with QnA dataset (360 samples each)

## Intended Use

This dataset is intended for:

1. **Training NER models** for multi-industry Lean Six Sigma entity extraction
2. **Fine-tuning language models** for domain-specific entity recognition across business verticals
3. **Developing knowledge extraction systems** for business process improvement across industries
4. **Educational applications** in comprehensive Lean Six Sigma methodology training
5. **Paired training** with the corresponding QnA dataset for comprehensive multi-industry understanding
6. **Cross-industry entity mapping** and best practice identification

## Model Performance

Recommended approaches:
- **Sequence labeling**: Use with BERT-based models for token classification across industries
- **Generative NER**: Fine-tune instruction-following models for multi-domain entity extraction
- **Multi-task learning**: Combine with QnA dataset for comprehensive Lean Six Sigma understanding
- **Domain adaptation**: Train on specific industries or use transfer learning across domains

## Entity Statistics

The dataset contains comprehensive entity coverage:
- **Total unique entities**: 2000+ across all industries and DMAIC phases
- **Average entities per sample**: 15-25 entities per sample
- **DMAIC phase distribution**: Balanced coverage across Define, Measure, Analyze, Improve, Control
- **Industry balance**: Equal representation (60 samples per major domain)
- **Modern techniques**: Integration of digital transformation, AI/ML, and automation entities

## Limitations

- Limited to 360 samples (optimized for few-shot learning and fine-tuning)
- English language only
- Requires domain expertise to evaluate entity categorization quality across different industries
- Entity categories limited to DMAIC framework phases
- Focus on operational and process improvement domains

## Citation

If you use this dataset in your research, please cite:

```
@dataset{lean_six_sigma_ner_2025,
  title={Lean Six Sigma NER Dataset},
  author={Clarence Wong},
  year={2025},
  url={https://huggingface.co/datasets/cw18/lean-six-sigma-ner},
  samples={360},
  domains={healthcare, ecommerce, manufacturing, energy_utilities, data_center_operations, supply_chain_logistics}
}
```

## License

This dataset is released under the MIT License, allowing for both commercial and non-commercial use.
"""

def prepare_dataset_for_upload(data):
    """Convert JSON data to Hugging Face Dataset format"""
    print("Preparing dataset for upload...")
    df = pd.DataFrame(data)
    print(f"Creating single train split with {len(data)} samples")
    train_dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({'train': train_dataset})
    return dataset_dict

def upload_to_huggingface(dataset_dict, repo_name, dataset_type, private=False):
    """Upload dataset to Hugging Face Hub"""
    print(f"Uploading {dataset_type} dataset to: {repo_name}")
    if dataset_type == 'QnA':
        card_content = create_qna_dataset_card()
    elif dataset_type == 'NER':
        card_content = create_ner_dataset_card()
    elif dataset_type == 'CoT':
        card_content = create_cot_dataset_card()
    else:
        card_content = ""
    
    # Set task categories based on dataset type
    if dataset_type == 'QnA':
        task_categories = ["question-answering", "text-generation"]
        tags = ["lean-six-sigma", "business-consulting", "process-improvement", 
               "supply-chain-logistics", "manufacturing", "quality-management", 
               "data-center-operations", "healthcare", "e-commerce", "energy-utilities", 
               "DMAIC", "instruction-following", "360-samples", "multi-domain", 
               "professional-training", "industrial-applications"]
    elif dataset_type == 'NER':
        task_categories = ["token-classification", "text-generation"]
        tags = ["lean-six-sigma", "business-consulting", "process-improvement", 
               "supply-chain-logistics", "manufacturing", "quality-management", 
               "data-center-operations", "healthcare", "e-commerce", "energy-utilities", 
               "DMAIC", "NER", "entity-extraction", "named-entity-recognition", 
               "360-samples", "multi-domain", "professional-training", "industrial-applications"]
    elif dataset_type == 'CoT':
        task_categories = ["text-generation", "question-answering"]
        tags = ["lean-six-sigma", "chain-of-thought", "chain-of-thought-reasoning", "business-consulting", 
                "process-improvement", "multi-domain", "DMAIC", "advanced-reasoning", 
                "statistical-analysis", "decision-support", "360-samples", 
                "professional-training", "industrial-applications"]
    
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
    print("3. CoT Dataset (Chain-of-Thought Reasoning Samples)")
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
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
        elif choice == '3':
            dataset_type = 'CoT'
            cot_dir = r"c:\Users\clwong\OneDrive\Documents\Learning\AI_MasterBlackBelt\datasets\lss_CoT"
            default_name = "lean-six-sigma-cot"
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    print(f"\n✅ Selected: {dataset_type} Dataset")
    if dataset_type in ['QnA', 'NER']:
        print(f"📁 Source file: {dataset_path}")
    elif dataset_type == 'CoT':
        print(f"📁 Source directory: {cot_dir}")
    
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
            exit(1)
    
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
        exit(1)
    
    # Load and validate dataset
    try:
        if dataset_type in ['QnA', 'NER']:
            data = load_and_validate_dataset(dataset_path, dataset_type)
        elif dataset_type == 'CoT':
            data = load_and_validate_cot_samples(cot_dir)
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path if dataset_type != 'CoT' else cot_dir}")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        exit(1)
    
    # Prepare dataset
    try:
        dataset_dict = prepare_dataset_for_upload(data)
        print("✅ Dataset prepared successfully!")
    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")
        exit(1)
    
    # Upload to Hugging Face
    print(f"\n📤 Uploading to Hugging Face Hub...")
    success = upload_to_huggingface(dataset_dict, repo_name, dataset_type, private)
    
    if success:
        print("\n🎉 Upload completed successfully!")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{repo_name}")
        print(f"📚 You can now use: load_dataset('{repo_name}')")
    else:
        print("\n❌ Upload failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
