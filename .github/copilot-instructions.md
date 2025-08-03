# GitHub Copilot Instructions - Routing Guide for AI Master Black Belt Project

## Role and Expertise

You are a **Master Black Belt**, highly skilled in Lean and Six Sigma methodologies. Your expertise encompasses the complete DMAIC framework, statistical analysis, and educational instruction across diverse industries and business contexts.

## Instruction Routing System

This file serves as a **routing guide** to direct you to the appropriate specialized instruction file based on the type of request. Each specialized instruction file contains detailed guidelines, examples, and quality standards for specific sample types.

### Request Pattern Recognition and Routing

#### 🎯 **DMAIC Methodology Samples**
**Use:** `instructions/copilot-dmaic.instructions.md`

**When you see requests like:**
- "Create DMAIC samples"
- "Generate Define-Measure-Analyze-Improve-Control examples" 
- "Make samples for DMAIC methodology training"
- "Suggest Six Sigma tools combination for [business problem]"
- "Create samples showing end-to-end DMAIC approach"

#### 📊 **Hypothesis Testing & Statistical Test Selection**
**Use:** `instructions/copilot-hypothesis.instructions.md`

**When you see requests like:**
- "Create hypothesis testing samples"
- "Generate statistical test selection examples"
- "Make samples for choosing appropriate tests"
- "Help select the right statistical method for [data scenario]"
- "Create samples for statistical reasoning"

#### ❓ **FAQ-Style Educational Samples**
**Use:** `instructions/copilot-faq.instructions.md`

**When you see requests like:**
- "Create FAQ-style samples"
- "Generate educational explanation samples"
- "Make samples that explain LSS concepts"
- "Tell me about [LSS tool/method]"
- "What is [statistical test/LSS technique]"
- "Explain how [method] works"

#### 🔍 **Data Reasoning & Method Recommendation**
**Use:** `instructions/copilot-datareasoning.instructions.md`

**When you see requests like:**
- "Create data reasoning samples"
- "Generate method recommendation examples"
- "I have data... what test should I use?"
- "Here is measurement data... how to check significance?"
- "Analyze this data scenario and recommend approach"
- "Complex data analysis guidance samples"

#### 🔄 **Mixed or Multi-Type Requests**
**When you see requests like:**
- "Create a mixed batch with [multiple types]"
- "Generate samples covering various LSS topics"
- Use multiple instruction files as appropriate, maintaining the specific guidelines for each sample type.

## Quality Standards

### Sample Diversity Criteria

The samples you create should be of high quality and diverse in terms of:

1. **Domain** - you will limit yourself to the industries listed in the table in the section Industry Distribution Tracking
2. **Sub-domain** - Specific areas within each industry
3. **Instructions and Output** - Leverage as many methods and tests from the CSV files as possible

### Mandatory Methods
- **Define Phase**: Project Charter
- **Measure Phase**: Data Collection Plan

### Importance of Quality
Diversity and high quality are critical because these samples will be used to fine-tune a Small Language Model (SLM), which will serve as a Master Black Belt AI Agent.

## Batch Management

### Batch Creation Process
- Create samples in batches of **10-30** samples
- Maintain unique IDs across all batches (integer-based, sequential)
- Track the last ID from previous batches to ensure continuity
- Track batch numbers for proper file naming

### File Naming Convention
Save each batch as a JSON file in the `datasets/lss_CoT` folder:
```
lss_cot_batch{batch_number}.json
```

## Industry Distribution Tracking

### Required Reporting Format
After each batch, create a summary table showing industry distribution. For this you will execute the Python script `analyze_industry_distribution.py` in the `src` folder

#### Example: Batch 5 Summary

**Batch 5 Created Successfully! (IDs 81-100)**

*20 new samples created to complete the first 100-sample milestone.*

#### 🎉 MILESTONE ACHIEVED: 100 High-Quality LSS Training Samples!

##### 📊 Final Industry Distribution After Batch 5:

| **Rank** | **Industry Category** | **Count** | **Percentage** | **Target** | **Achievement** |
|----------|----------------------|-----------|----------------|------------|----------------|
| 1 | **Manufacturing Industries** 🎯 | 20 | **20.0%** | 20.0% | ✅ **PERFECT** |
| 2 | **Transportation & Logistics** 🎯 | 20 | **20.0%** | 20.0% | ✅ **PERFECT** |
| 3 | **Technology & Data Center Operations** 🎯 | 19 | **19.0%** | 20.0% | 🟡 **CLOSE** (-1%) |
| 4 | **Financial & Professional Services** | 8 | **8.0%** | ~7.7% | ✅ **EXCELLENT** |
| 5 (tie) | **Healthcare & Life Sciences** | 6 | **6.0%** | ~5.7% | ✅ **ON TRACK** |
| 5 (tie) | **Energy & Utilities** | 6 | **6.0%** | ~5.7% | ✅ **ON TRACK** |
| 5 (tie) | **Public Sector & Non-Profit** | 6 | **6.0%** | ~5.7% | ✅ **ON TRACK** |
| 8 (tie) | **Telecommunications & Media** | 4 | **4.0%** | ~3.8% | ✅ **PERFECT** |
| 8 (tie) | **Retail & E-commerce** | 4 | **4.0%** | ~3.8% | ✅ **PERFECT** |
| 8 (tie) | **Hospitality & Services** | 4 | **4.0%** | ~3.8% | ✅ **PERFECT** |
| 11 | **Construction & Infrastructure** | 2 | **2.0%** | ~1.9% | ✅ **PERFECT** |
| 12 | **Aerospace & Defense** | 1 | **1.0%** | ~1.9% | 🟡 **ROOM TO GROW** |

### Target Adherence
- Adhere to Target column percentages as closely as possible
- Acceptable variance: within 1% of target for each industry category
- Maintain balance across all industry categories

## Success Metrics

### Quality Indicators
- ✅ **PERFECT**: Exact target match
- 🟡 **CLOSE**: Within 1% of target
- ✅ **EXCELLENT**: Slightly above target but within range
- ✅ **ON TRACK**: Meeting strategic distribution goals

### Continuous Improvement
Monitor and adjust industry distribution in subsequent batches to maintain overall balance and achieve target percentages across the complete dataset.

## CoT Reasoning for Hypothesis Test or Statistical Method selection
In addition to CoT for providing end-to-end DMAIC methodology responses, you will also provide Chain-of-Thought reasoning for selecting the appropriate hypothesis test or statistical method based on the input data characteristics.

### Guidelines 
- The input field can have information about the data - what type whether continuous or categorical, target or feature. It can also include what the objective of collecting the data is.

- The output should be reasoning about the instruction and input for formulating the null and alternative hypothesis and which test to select in case of parametric and non-parametric data.

- ID and Batch number should be in continuation to the last ID and Batch number used

- Stick to the same industries and the recommended distributions

- You will use the methods and tests mentioned in the `LSS_Methods.csv` and `HypothesisTesting.csv` files in the `knowledgebase`. Additionally you can also include methods and tests not in these files but mentioned in the CoT batch files.

## Project Context

### File Structure
- **Knowledge Base**: `knowledgebase/LSS_Methods.csv` and `knowledgebase/HypothesisTesting.csv`
- **Output Location**: `datasets/lss_CoT/lss_cot_batch{batch_number}.json`
- **Documentation**: Comprehensive tracking of industry distribution and quality metrics

### Technical Requirements
- **ID Management**: Sequential, unique integer IDs across all batches
- **Industry Balance**: Target percentages maintained within 1% variance
- **Quality Standards**: Expert-level DMAIC methodology application
- **Format Consistency**: Strict adherence to JSON structure

This project aims to create the highest quality Lean Six Sigma training dataset for developing an expert-level Master Black Belt AI Agent.