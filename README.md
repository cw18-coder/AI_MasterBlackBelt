# AI Master Black Belt - Lean Six Sigma Dataset Project

🎯 **Comprehensive Lean Six Sigma datasets for fine-tuning Small Language Models (SLMs)**

This repository contains curated Lean Six Sigma datasets and tools for training AI models to provide expert-level business process improvement consulting.

## 🚀 Project Overview

This project focuses on creating high-quality, domain-specific datasets for training Small Language Models (3B-8B parameters) to become specialized Lean Six Sigma consultants. The datasets cover supply chain optimization, manufacturing processes, and business improvement methodologies following the DMAIC framework.

### Key Features

- **102 aligned QnA and NER samples** covering diverse Lean Six Sigma scenarios
- **Supply chain focus** with enhanced logistics and manufacturing coverage
- **DMAIC methodology alignment** (Define, Measure, Analyze, Improve, Control)
- **Ready-to-use upload scripts** for Hugging Face Datasets Hub
- **Secure token management** with .env integration
- **Professional documentation** and usage examples

## 📊 Datasets

### QnA Dataset (`sixSigma_QnA_caseStudy_sample.json`)
- **102 question-answer pairs** for instruction-following model fine-tuning
- Expert-level responses following DMAIC methodology
- Real-world business scenarios and case studies
- Alpaca format compatibility for training

### NER Dataset (`sixSigma_NER_caseStudy_sample.json`)
- **102 Named Entity Recognition samples** for entity extraction
- DMAIC phase categorization of Lean Six Sigma tools and methodologies
- Aligned with QnA dataset for comprehensive training
- Supports both token classification and generative NER approaches

### Coverage Areas

#### Supply Chain & Logistics
- Material handling optimization
- Supply chain visibility enhancement
- Production planning improvement
- Cold chain logistics management
- Cross-docking operations
- Reverse logistics optimization
- Last-mile delivery enhancement
- Route optimization
- Order fulfillment efficiency

#### Quality & Process Improvement
- Cycle time reduction
- Flow optimization
- Supplier quality management
- Demand forecasting accuracy
- Procurement efficiency
- Distribution optimization
- Warehouse productivity
- Inventory management
- Freight optimization

#### Specialized Areas
- Sustainable supply chain practices
- Trade compliance optimization
- Supply chain resilience building

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+ (tested with 3.11.8)
- Git (for cloning and version control)
- Hugging Face account with write token

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AI_MasterBlackBelt
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip --no-cache-dir
   pip install -r requirements_upload.txt --upgrade --no-cache-dir
   ```

4. **Configure Hugging Face credentials**
   ```bash
   cp .env.template .env
   # Edit .env with your Hugging Face token and username
   ```

5. **Upload datasets**
   ```bash
   python src/upload_to_huggingface.py
   ```

## 📁 Repository Structure

```
AI_MasterBlackBelt/
├── README.md                          # This file
├── .env.template                      # Environment variables template
├── .gitignore                         # Git ignore rules
├── requirements_upload.txt            # Python dependencies
├── SETUP_TOKEN.md                     # Hugging Face setup guide
├── src/
│   └── upload_to_huggingface.py      # Main upload script
├── datasets/
│   ├── lss_consultant/
│   │   └── sixSigma_QnA_caseStudy_sample.json
│   └── lss_ner/
│       └── sixSigma_NER_caseStudy_sample.json
└── ChatGPT/                          # Dataset curation history
    └── DatasetCuration/
        ├── Iteration1/
        ├── Iteration2/
        └── Iteration3/
```

## 🔧 Usage

### Uploading Datasets to Hugging Face

The main script provides an interactive interface for uploading datasets:

```bash
python src/upload_to_huggingface.py
```

**Features:**
- ✅ Interactive dataset selection (QnA or NER)
- ✅ Automatic token authentication from .env
- ✅ Default username from environment variables
- ✅ Comprehensive README generation
- ✅ Error handling and validation
- ✅ Private/public repository options

### Using the Datasets

#### Loading from Hugging Face Hub

```python
from datasets import load_dataset

# Load QnA dataset
qna_dataset = load_dataset("cw18/lean-six-sigma-qna")['train']

# Load NER dataset  
ner_dataset = load_dataset("cw18/lean-six-sigma-ner")['train']
```

#### Training Data Preparation

```python
# Alpaca format for QnA
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
formatted_dataset = qna_dataset.map(lambda x: {"text": format_alpaca_prompt(x)})
```

## 🤖 Model Recommendations

### Optimal Models for Fine-tuning

| Model | Parameters | VRAM | Training Time | Use Case |
|-------|------------|------|---------------|----------|
| **Llama 3.2 3B** | 3B | 6GB | 2-3 hours | Balanced performance/efficiency |
| **Mistral 7B** | 7B | 16GB | 1.5-2 hours | Excellent instruction following |
| **Qwen 2.5 7B** | 7B | 16GB | 1-1.5 hours | Strong reasoning capabilities |
| **Gemma 7B** | 7B | 16GB | 2-2.5 hours | Google's instruction-tuned model |

### Training Frameworks
- **Unsloth**: Parameter-efficient fine-tuning with LoRA/QLoRA
- **Hugging Face Transformers**: Full fine-tuning capabilities
- **Axolotl**: Advanced training configurations

## 🔒 Security & Best Practices

### Environment Variables
- ✅ All sensitive tokens stored in `.env` file
- ✅ `.env` excluded from version control
- ✅ Template provided for easy setup
- ✅ Multiple token name support

### Repository Hygiene
- ✅ Comprehensive `.gitignore` for ML projects
- ✅ Virtual environments excluded
- ✅ Dataset files and checkpoints ignored
- ✅ Jupyter notebook checkpoints excluded

## 📈 Performance Metrics

### Dataset Quality
- **Expert validation**: All responses follow proper DMAIC methodology
- **Real-world scenarios**: Based on actual business challenges
- **Comprehensive coverage**: 20+ sub-domains across supply chain and quality
- **Alignment**: Perfect ID and structure alignment between QnA and NER datasets

### Training Performance
- **Parameter efficiency**: Optimized for LoRA/QLoRA training
- **Data efficiency**: 102 samples suitable for few-shot learning
- **Domain specificity**: Focused on business process improvement
- **Practical application**: Ready for production consulting scenarios

## 🤝 Contributing

### Dataset Enhancement
1. Follow the existing JSON structure for new samples
2. Maintain ID alignment between QnA and NER datasets
3. Include proper DMAIC methodology in responses
4. Test with the upload script before submission

### Code Improvements
1. Fork the repository
2. Create a feature branch
3. Test all changes thoroughly
4. Submit a pull request with detailed description

## 📄 License

This project is released under the **MIT License**, allowing for both commercial and non-commercial use.

### Citation

If you use these datasets in your research or projects, please cite:

```bibtex
@dataset{lean_six_sigma_datasets_2025,
  title={Lean Six Sigma Datasets for AI Model Training},
  author={Clarence Wong},
  year={2025},
  url={https://github.com/cw18/ai-master-black-belt},
  note={QnA and NER datasets for business process improvement}
}
```

## 🔗 Links

- **Hugging Face QnA Dataset**: [cw18/lean-six-sigma-qna](https://huggingface.co/datasets/cw18/lean-six-sigma-qna)
- **Hugging Face NER Dataset**: [cw18/lean-six-sigma-ner](https://huggingface.co/datasets/cw18/lean-six-sigma-ner)
- **Setup Guide**: [SETUP_TOKEN.md](SETUP_TOKEN.md)

## 📞 Support

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: Use the repository issue tracker
- **Dataset Questions**: Check the Hugging Face dataset pages
- **Training Support**: Refer to model-specific documentation

---

**🎯 Ready to train your Lean Six Sigma AI consultant!**

Transform your business process improvement with AI-powered expertise built on comprehensive, real-world datasets.
