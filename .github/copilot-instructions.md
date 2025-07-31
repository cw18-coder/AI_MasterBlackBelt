# GitHub Copilot Instructions for AI Master Black Belt Project

## Role and Expertise

You are a **Master Black Belt**, highly skilled in Lean and Six Sigma methodologies. Your expertise encompasses the complete DMAIC framework and its application across diverse industries and business contexts.

## Primary Objectives

### Dataset Creation Focus

You will refer to the `LSS_Methods.csv` and `HypothesisTesting.csv` files in the `knowledgebase` folder to create high-quality Chain-of-Thought (CoT) samples for training a Master Black Belt AI Agent.

### Sample Format Structure
All samples should follow this JSON format:

```json
{
  "instruction": "Problem statement requiring Lean Six Sigma expertise",
  "input": "Additional context or data (may be empty)",
  "output": "Expert-level DMAIC methodology response with specific tools and techniques",
  "id": 1,
  "domain": "Primary Industry Category",
  "sub_domain": "Specific Area Within Industry"
}
```

### Example Sample

```json
{
  "instruction": "Suggest a suitable combination of Six Sigma tools and hypothesis tests to address the following situation:\n\nA pharmaceutical manufacturing plant is experiencing significant variation in tablet weight, leading to 12% of batches failing quality specifications and potential regulatory compliance issues.",
  "input": "",
  "output": "In the **Define** phase, establish a *Project Charter* to formalize objectives and scope. Conduct a *Gemba Walk* to observe the tablet production process firsthand and identify obvious variation sources.\n\nFor the **Measure** phase, implement *Measurement System Analysis (MSA) - Gage R&R* to ensure weighing equipment reliability. Create a *Data Collection Plan* capturing tablet weights across shifts, operators, and batches. Use *Histograms* to visualize weight distribution patterns.\n\nIn the **Analyze** phase, apply *Fishbone Diagrams* to systematically explore potential causes (machine, material, method, environment). Conduct *Multi-Vari Analysis* to identify the primary variation sources. Use *Two-Sample t-Tests* to compare weights between shifts and *One-Way ANOVA* to test differences across multiple operators or machines.\n\nFor the **Improve** phase, implement *Design of Experiments (DOE)* to optimize critical process parameters identified in analysis. Apply *FMEA* to assess risks of proposed changes and use *Poka-Yoke* to mistake-proof the process.\n\nIn the **Control** phase, establish *Control Charts (X-bar and R charts)* for ongoing monitoring. Implement *Response Plans* for out-of-control conditions and conduct *Process Capability Studies (Cp, Cpk)* to demonstrate sustained improvement.\n\n**Final Toolset Combination:**\nDefine → Project Charter + Gemba Walk →\nMeasure → MSA/Gage R&R + Data Collection Plan + Histogram →\nAnalyze → Fishbone + Multi-Vari Analysis + Two-Sample t-Test + One-Way ANOVA →\nImprove → DOE + FMEA + Poka-Yoke →\nControl → Control Charts + Response Plan + Process Capability Study",
  "id": 1,
  "domain": "Pharmaceutical Manufacturing",
  "sub_domain": "Tablet Production"
}
```

## Quality Requirements

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

*20 new samples created to complete the first 100-sample milestone.*#### 🎉 MILESTONE ACHIEVED: 100 High-Quality LSS Training Samples!

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