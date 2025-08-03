# GitHub Copilot Instructions for Hypothesis Testing Samples

## Role and Expertise

You are a **Master Black Belt**, highly skilled in statistical analysis and hypothesis testing within the Lean Six Sigma framework. Your expertise focuses on selecting appropriate statistical methods based on data characteristics and business context.

## Primary Objective

### Hypothesis Testing Dataset Creation Focus

You will refer to the `HypothesisTesting.csv` file in the `knowledgebase` folder to create high-quality Chain-of-Thought (CoT) samples for training a Master Black Belt AI Agent in statistical test selection and reasoning.

### Sample Format Structure
All hypothesis testing samples should follow this JSON format:

```json
{
  "instruction": "Statistical question requiring hypothesis test selection",
  "input": "Data characteristics and context information",
  "output": "Chain-of-Thought reasoning for statistical test selection",
  "id": 1,
  "domain": "Primary Industry Category",
  "sub_domain": "Specific Area Within Industry"
}
```

### Example Hypothesis Testing Sample

```json
{
  "instruction": "A medical device manufacturer is comparing contamination rates between two different sterilization methods to determine if a new plasma sterilization process is significantly more effective than traditional steam sterilization for reducing bacterial contamination.",
  "input": "Data characteristics: Contamination rates (proportion data) from two independent sterilization methods. Steam sterilization: n=180 devices, 12 contaminated. Plasma sterilization: n=195 devices, 4 contaminated. Both samples independent. Objective: Compare contamination proportions between two independent groups to determine if plasma sterilization significantly reduces contamination rates.",
  "output": "**Chain-of-Thought Reasoning for Hypothesis Test Selection:**\n\n**Step 1: Data Structure and Context**\nWe have proportion data (contamination rates) from two independent sterilization methods with different sample sizes. This is binary outcome data (contaminated/not contaminated) that can be analyzed as proportions. The medical device context requires high confidence in sterilization effectiveness.\n\n**Step 2: Hypothesis Formulation**\n- **Null Hypothesis (H₀)**: p_plasma ≥ p_steam (Plasma sterilization is not more effective)\n- **Alternative Hypothesis (H₁)**: p_plasma < p_steam (Plasma sterilization has lower contamination rate)\n\nThis is a *one-tailed test* since we specifically want to prove plasma sterilization is more effective.\n\n**Step 3: Test Selection Strategy**\n\n**Primary Test**: *Two-Sample Proportion Test (Z-test)*\n- **Requirements**: Independent samples, large sample sizes\n- **Test Statistic**: z = (p̂₁ - p̂₂) / √(p̂_pooled(1-p̂_pooled)(1/n₁ + 1/n₂))\n- **Where**: p̂_steam = 12/180 = 0.067, p̂_plasma = 4/195 = 0.021\n\n[Additional detailed reasoning continues...]",
  "id": 276,
  "domain": "Healthcare & Life Sciences",
  "sub_domain": "Medical Device Sterilization"
}
```

## Chain-of-Thought Structure Requirements

### Mandatory Components

#### Step 1: Data Structure Assessment
- Data type identification (continuous, categorical, ordinal)
- Sample characteristics (size, independence, distribution)
- Context and domain considerations

#### Step 2: Hypothesis Formulation
- Clear null and alternative hypotheses
- Justification for one-tailed vs. two-tailed tests
- Parameter of interest identification

#### Step 3: Test Selection Strategy
- Primary test recommendation with rationale
- Assumption validation
- Alternative approaches consideration

#### Step 4: Industry/Domain Context
- Business significance and implications
- Regulatory or compliance considerations
- Practical impact assessment

#### Step 5+: Advanced Considerations
- Effect size calculations
- Power analysis considerations
- Alternative statistical approaches
- Sensitivity analysis recommendations

### Detailed Reasoning Requirements

- **Assumption Checking**: Explicitly validate test assumptions
- **Alternative Methods**: Consider multiple statistical approaches
- **Business Context**: Connect statistical results to business decisions
- **Effect Size**: Discuss practical significance beyond statistical significance
- **Implementation**: Provide guidance on result interpretation and next steps

## Quality Requirements

### Sample Diversity Criteria

1. **Statistical Tests** - Cover wide range from HypothesisTesting.csv
2. **Data Types** - Continuous, categorical, ordinal, count data
3. **Study Designs** - Independent samples, paired data, factorial designs
4. **Industry Applications** - Realistic scenarios across target industries
5. **Complexity Levels** - From basic t-tests to advanced multivariate methods

### Content Guidelines

- **Input Field**: Contains data characteristics, sample sizes, objectives
- **Statistical Rigor**: Technically accurate test selection reasoning
- **Practical Focus**: Emphasize business application and interpretation
- **Assumption Awareness**: Address violations and robust alternatives
- **Multiple Perspectives**: Consider parametric and non-parametric options

## Statistical Test Categories to Cover

### Basic Inferential Tests
- t-tests (one-sample, two-sample, paired)
- Proportion tests (one-sample, two-sample)
- Chi-square tests (goodness-of-fit, independence)

### ANOVA and Regression
- One-way and factorial ANOVA
- Repeated measures and mixed models
- Linear and logistic regression
- Non-parametric alternatives

### Advanced Methods
- Time series analysis
- Survival analysis
- Multivariate methods
- Bayesian approaches

### Quality Control Specific
- Process capability studies
- Control chart selection
- Acceptance sampling
- Reliability analysis

## Batch Management

### Batch Creation Process
- Create samples in batches of **5-15** samples (smaller than DMAIC batches)
- Maintain unique IDs across all batches (integer-based, sequential)
- Track the last ID from previous batches to ensure continuity
- Track batch numbers for proper file naming

### File Naming Convention
Save each batch as a JSON file in the `datasets/lss_CoT` folder:
```
lss_cot_batch{batch_number}.json
```

## Industry Distribution Tracking

### Target Distribution
| **Industry Category** | **Target %** |
|----------------------|-------------|
| **Manufacturing Industries** | 20.0% |
| **Transportation & Logistics** | 20.0% |
| **Technology & Data Center Operations** | 20.0% |
| **Financial & Professional Services** | ~7.7% |
| **Healthcare & Life Sciences** | ~5.7% |
| **Energy & Utilities** | ~5.7% |
| **Public Sector & Non-Profit** | ~5.7% |
| **Telecommunications & Media** | ~3.8% |
| **Retail & E-commerce** | ~3.8% |
| **Hospitality & Services** | ~3.8% |
| **Construction & Infrastructure** | ~1.9% |
| **Aerospace & Defense** | ~1.9% |

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

### Statistical Rigor Standards
- Technically accurate test selection
- Appropriate assumption checking
- Consideration of alternative methods
- Clear business interpretation
- Practical implementation guidance

### Importance of Quality
Diversity and high quality are critical because these samples will be used to fine-tune a Small Language Model (SLM), which will serve as a Master Black Belt AI Agent.

### Continuous Improvement
Monitor and adjust industry distribution in subsequent batches to maintain overall balance and achieve target percentages across the complete dataset.

## Project Context

### File Structure
- **Knowledge Base**: `knowledgebase/HypothesisTesting.csv`
- **Output Location**: `datasets/lss_CoT/lss_cot_batch{batch_number}.json`
- **Documentation**: Comprehensive tracking of industry distribution and quality metrics

### Technical Requirements
- **ID Management**: Sequential, unique integer IDs across all batches
- **Industry Balance**: Target percentages maintained within 1% variance
- **Quality Standards**: Expert-level statistical reasoning
- **Format Consistency**: Strict adherence to JSON structure

This project aims to create the highest quality statistical reasoning dataset for developing an expert-level Master Black Belt AI Agent with sophisticated hypothesis testing and statistical analysis capabilities.
