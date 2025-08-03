# GitHub Copilot Instructions for FAQ-Style CoT Samples

## Role and Expertise

You are a **Master Black Belt**, highly skilled in Lean and Six Sigma methodologies with deep expertise in explaining concepts, methods, and their applications. Your role is to provide clear, educational responses to frequently asked questions about LSS tools, techniques, and statistical methods.

## Primary Objective

### FAQ Dataset Creation Focus

You will refer to both `LSS_Methods.csv` and `HypothesisTesting.csv` files in the `knowledgebase` folder to create high-quality Chain-of-Thought (CoT) samples that answer common questions about Lean Six Sigma methods, statistical tests, and their practical applications.

### Sample Format Structure
All FAQ samples should follow this JSON format:

```json
{
  "instruction": "FAQ-style question about LSS methods or statistical tests",
  "input": "Additional context or scenario (may be empty)",
  "output": "Educational Chain-of-Thought response explaining the concept and its application",
  "id": 1,
  "domain": "Primary Industry Category",
  "sub_domain": "Specific Area Within Industry"
}
```

## FAQ Question Categories

### Category 1: Method/Test Explanation Questions
**Pattern**: "Tell me about...", "What is...", "Explain...", "How does... work?"

#### Example 1A - Method Explanation
```json
{
  "instruction": "Tell me about Multi-Vari Analysis and when it should be used in Six Sigma projects.",
  "input": "",
  "output": "**Multi-Vari Analysis Overview:**\n\nMulti-Vari Analysis is a powerful statistical tool used in the **Analyze** phase of DMAIC to identify the primary sources of variation in a process by systematically examining variation patterns across different factors.\n\n**Key Principles:**\n- **Positional variation**: Variation within a single unit or sample\n- **Cyclical variation**: Variation between consecutive units in time sequence\n- **Temporal variation**: Variation between different time periods, shifts, or batches\n\n**When to Use Multi-Vari Analysis:**\n1. **Process variation investigation**: When you need to identify which factors contribute most to process variability\n2. **Root cause prioritization**: To focus improvement efforts on the biggest variation drivers\n3. **DOE preparation**: Before designing experiments, to understand which factors matter most\n4. **Complex processes**: When multiple potential sources of variation exist\n\n**Practical Application Steps:**\n1. **Data stratification**: Organize data by potential variation sources (time, location, operator, material, etc.)\n2. **Graphical analysis**: Create charts showing variation patterns across different factors\n3. **Variance component analysis**: Calculate percentage contribution of each factor\n4. **Prioritization**: Focus on factors contributing >80% of total variation\n\n**Industry Applications:**\n- **Manufacturing**: Identifying machine, operator, or material lot effects\n- **Healthcare**: Analyzing variation in treatment outcomes across providers, time, or patient groups\n- **Service**: Understanding variation in process times across different service channels\n\n**Integration with Other Tools:**\n- Follows **Fishbone Diagram** for hypothesis generation\n- Precedes **Design of Experiments** for factor optimization\n- Supports **Control Chart** subgroup selection\n- Complements **Statistical Process Control** implementation",
  "id": 301,
  "domain": "Manufacturing Industries",
  "sub_domain": "Quality Control"
}
```

#### Example 1B - Statistical Test Explanation
```json
{
  "instruction": "What is the Mann-Whitney U Test and when should I use it instead of a t-test?",
  "input": "",
  "output": "**Mann-Whitney U Test Overview:**\n\nThe Mann-Whitney U Test (also called Wilcoxon Rank-Sum Test) is a non-parametric statistical test used to compare two independent groups when data doesn't meet the assumptions required for parametric tests like the t-test.\n\n**Key Characteristics:**\n- **Non-parametric**: Doesn't assume normal distribution\n- **Rank-based**: Uses ranks rather than actual values\n- **Robust**: Less sensitive to outliers and skewed data\n- **Distribution-free**: Works with any continuous distribution shape\n\n**When to Use Mann-Whitney U Instead of t-Test:**\n\n**1. Violated Normality Assumptions:**\n- Data is heavily skewed (right or left)\n- Small sample sizes with non-normal distributions\n- Presence of extreme outliers affecting normality\n\n**2. Ordinal Data:**\n- Customer satisfaction ratings (1-5 scale)\n- Pain scores in medical studies\n- Quality rankings or ratings\n\n**3. Unknown Distributions:**\n- When you can't verify distributional assumptions\n- New processes without historical data patterns\n- Mixed data types or measurement scales\n\n**Practical Decision Framework:**\n\n**Use t-Test When:**\n- ✅ Data is approximately normal (Shapiro-Wilk p > 0.05)\n- ✅ Sample sizes are adequate (n ≥ 30 per group)\n- ✅ Variances are reasonably equal\n- ✅ Continuous measurement scale\n\n**Use Mann-Whitney U When:**\n- ❌ Normality violated (Shapiro-Wilk p < 0.05)\n- ❌ Severe outliers present\n- ❌ Small sample sizes with questionable normality\n- ❌ Ordinal or ranked data\n- ❌ Unequal variances with non-normal data\n\n**Industry Applications:**\n- **Manufacturing**: Comparing defect rates between shifts with skewed data\n- **Healthcare**: Analyzing patient wait times (typically right-skewed)\n- **Service**: Comparing customer satisfaction scores between locations\n- **Quality Control**: Evaluating process performance with outlier-prone data\n\n**Interpretation Considerations:**\n- Tests for **stochastic dominance** rather than mean differences\n- Results indicate probability that one group tends to have higher values\n- Effect size measured using rank-biserial correlation or Cliff's delta\n- More conservative than t-test but provides robust conclusions",
  "id": 302,
  "domain": "Healthcare & Life Sciences",
  "sub_domain": "Clinical Data Analysis"
}
```

### Category 2: Application/Suitability Questions
**Pattern**: "Is...", "Will...", "Would...", "Can...", "Do you think...", "Should I..."

#### Example 2A - Method Suitability
```json
{
  "instruction": "Can I use Design of Experiments (DOE) for optimizing our software testing process, or is it only for manufacturing?",
  "input": "We're a software development company struggling with inconsistent bug detection rates across different testing approaches. We want to optimize our testing strategy by examining factors like test case design method, automation level, reviewer experience, and testing environment.",
  "output": "**Yes, Design of Experiments (DOE) is excellent for software testing optimization!** DOE is not limited to manufacturing and can be highly effective for software processes.\n\n**Why DOE Works for Software Testing:**\n\n**1. Multiple Factor Investigation:**\nYour scenario involves several factors that could interact:\n- **Test case design method**: Boundary value, equivalence partitioning, exploratory\n- **Automation level**: Manual, semi-automated, fully automated\n- **Reviewer experience**: Junior, intermediate, senior\n- **Testing environment**: Development, staging, production-like\n\n**2. Response Variable Optimization:**\nDOE can optimize multiple responses simultaneously:\n- **Primary**: Bug detection rate (defects found/total defects)\n- **Secondary**: Testing efficiency (bugs found/hour)\n- **Tertiary**: False positive rate (invalid bugs reported)\n\n**Recommended DOE Approach:**\n\n**Phase 1 - Screening Design (2^4-1 Fractional Factorial):**\n- Identify which factors have the biggest impact\n- 8 experimental runs to test main effects\n- Cost-effective initial investigation\n\n**Phase 2 - Optimization Design (Response Surface):**\n- Focus on significant factors from Phase 1\n- Find optimal factor level combinations\n- Include interaction effects analysis\n\n**Practical Implementation:**\n\n**Factor Level Selection:**\n- **Test case method**: Traditional vs. Risk-based vs. Exploratory\n- **Automation**: 0% vs. 50% vs. 100% automated\n- **Reviewer experience**: <2 years vs. 2-5 years vs. >5 years\n- **Environment**: Simplified vs. Production-like vs. Actual production data\n\n**Response Measurement:**\n- **Bug detection effectiveness**: Bugs found in testing / Total bugs found (including production)\n- **Efficiency metric**: Bugs found per testing hour\n- **Quality metric**: Valid bugs / Total bugs reported\n\n**Software Industry DOE Applications:**\n- **Code review optimization**: Reviewer assignment, review duration, checklist usage\n- **Development methodology**: Agile parameters, sprint length, team size\n- **Performance testing**: Load patterns, test duration, environment configuration\n- **User interface testing**: Screen resolution, browser type, user scenario complexity\n\n**Implementation Considerations:**\n\n**Advantages in Software:**\n- **Rapid experimentation**: Software changes are typically faster than manufacturing\n- **Controlled environment**: Better control over experimental conditions\n- **Measurable outcomes**: Clear metrics like defect rates, performance measures\n- **Reversible changes**: Easy to revert unsuccessful experimental conditions\n\n**Challenges to Address:**\n- **Standardization**: Ensure consistent bug classification criteria\n- **Randomization**: Properly randomize test case assignments\n- **Replication**: Use multiple projects or modules for replication\n- **Confounding factors**: Control for code complexity, developer experience\n\n**Expected Benefits:**\n- **15-30% improvement** in bug detection rates (typical for well-designed software DOE)\n- **Reduced testing time** through optimized factor combinations\n- **Data-driven decisions** rather than intuition-based testing strategies\n- **Scalable insights** applicable across multiple projects\n\nDOE will provide statistically valid evidence for optimizing your testing strategy, making it a powerful tool for software quality improvement.",
  "id": 303,
  "domain": "Technology & Data Center Operations",
  "sub_domain": "Software Quality Assurance"
}
```

#### Example 2B - Statistical Test Suitability
```json
{
  "instruction": "Would Chi-Square Test of Independence be appropriate for analyzing the relationship between customer payment method and return rates in our e-commerce business?",
  "input": "We have categorical data on payment methods (credit card, debit card, PayPal, digital wallet) and return status (returned vs. not returned) for 15,000 orders over the past quarter.",
  "output": "**Yes, Chi-Square Test of Independence is highly appropriate for this analysis!** Your scenario perfectly matches the requirements for this test.\n\n**Why Chi-Square Test of Independence Fits:**\n\n**1. Data Type Requirements Met:**\n- ✅ **Two categorical variables**: Payment method (4 categories) and return status (2 categories)\n- ✅ **Independent observations**: Each order is independent\n- ✅ **Adequate sample size**: 15,000 orders provides excellent power\n- ✅ **Mutually exclusive categories**: Each order has one payment method and one return outcome\n\n**2. Research Question Alignment:**\nThe test directly answers: \"Is there a significant association between payment method and return behavior?\"\n\n**Test Setup:**\n\n**Null Hypothesis (H₀):** Payment method and return rates are independent (no association)\n**Alternative Hypothesis (H₁):** Payment method and return rates are associated\n\n**Contingency Table Structure:**\n```\n                    Returned    Not Returned    Total\nCredit Card           a₁           b₁          n₁\nDebit Card            a₂           b₂          n₂\nPayPal                a₃           b₃          n₃\nDigital Wallet        a₄           b₄          n₄\nTotal               R_total      NR_total      15,000\n```\n\n**Assumption Validation:**\n\n**✅ Expected Cell Count Rule:**\n- With 15,000 orders, all expected frequencies will likely exceed 5\n- Minimum expected count = (row total × column total) / grand total\n- Your large sample size virtually guarantees this assumption is met\n\n**✅ Independence Assumption:**\n- Orders are independent events\n- No customer appears multiple times (or account for clustering if they do)\n\n**Analysis Approach:**\n\n**1. Descriptive Analysis:**\n- Calculate return rates for each payment method\n- Create crosstab with percentages\n- Visualize with stacked bar charts\n\n**2. Statistical Testing:**\n- Compute Chi-Square statistic: χ² = Σ[(Observed - Expected)²/Expected]\n- Degrees of freedom = (rows - 1) × (columns - 1) = 3 × 1 = 3\n- Compare to critical value or use p-value approach\n\n**3. Effect Size Assessment:**\n- **Cramér's V**: √(χ²/(n × min(r-1, c-1))) for practical significance\n- **Phi coefficient**: For 2×2 relationships (if you want to compare pairs)\n\n**Business Interpretation Framework:**\n\n**If Significant Association Found:**\n- **Identify patterns**: Which payment methods have higher/lower return rates?\n- **Customer behavior insights**: Do certain payment users behave differently?\n- **Fraud consideration**: Are higher return rates linked to payment security?\n- **Operational impact**: Focus customer service on high-return payment methods\n\n**Potential Follow-up Analyses:**\n\n**1. Post-hoc Pairwise Comparisons:**\n- Multiple 2×2 Chi-Square tests with Bonferroni correction\n- Identify which specific payment methods differ significantly\n\n**2. Stratified Analysis:**\n- Control for product category, price range, or customer demographics\n- Mantel-Haenszel test for stratified association\n\n**3. Logistic Regression:**\n- Include payment method as predictor variable\n- Add control variables (price, category, customer history)\n- Obtain odds ratios for business interpretation\n\n**E-commerce Specific Considerations:**\n\n**Practical Insights:**\n- **Payment friction**: Easier payment methods might correlate with impulsive purchases (higher returns)\n- **Customer segments**: Different payment preferences might indicate different customer types\n- **Fraud patterns**: Unusual return patterns by payment method might indicate fraudulent activity\n- **Customer confidence**: Secure payment methods might correlate with more confident purchases (lower returns)\n\n**Actionable Outcomes:**\n- **Marketing strategy**: Target low-return payment method users\n- **Product recommendations**: Customize based on payment method patterns\n- **Return policy**: Differentiate policies by payment method if legally permissible\n- **Customer experience**: Optimize checkout flow for high-return payment methods\n\n**Expected Results:**\nWith 15,000 orders, you'll have excellent statistical power to detect even small associations. The analysis will provide clear evidence about payment method-return relationships and actionable insights for business optimization.",
  "id": 304,
  "domain": "Retail & E-commerce",
  "sub_domain": "Customer Analytics"
}
```

### Category 3: Multi-Level Complex Questions
**Pattern**: Combination of explanation and application, often with multiple parts or scenarios

#### Example 3A - Complex Method Application
```json
{
  "instruction": "I understand FMEA is used to identify potential failures, but can you explain how it integrates with other Six Sigma tools in a real project, and would it be suitable for our airline's maintenance scheduling optimization?",
  "input": "We're an airline looking to optimize our aircraft maintenance scheduling to reduce unplanned downtime while ensuring safety compliance. Our current approach is reactive, and we're experiencing costly emergency repairs and flight delays.",
  "output": "**FMEA (Failure Mode and Effects Analysis) is excellent for airline maintenance optimization** and integrates powerfully with other Six Sigma tools in a comprehensive DMAIC approach.\n\n**FMEA Overview and Aviation Relevance:**\n\nFMEA systematically identifies potential failure modes, their causes, effects, and mitigation strategies. For airline maintenance, this means:\n- **Proactive risk management**: Identify failures before they occur\n- **Safety compliance**: Systematic approach meeting aviation regulations\n- **Cost optimization**: Balance maintenance costs with reliability\n- **Operational continuity**: Minimize unplanned aircraft downtime\n\n**FMEA Integration in DMAIC Framework:**\n\n**Define Phase Integration:**\n- **Project Charter** ← FMEA supports business case with risk quantification\n- **Voice of Customer** ← Passenger safety and schedule reliability requirements\n- **SIPOC** ← FMEA identifies failure points in maintenance processes\n\n**Measure Phase Integration:**\n- **Data Collection Plan** ← FMEA guides what failure data to collect\n- **Historical failure analysis** ← FMEA framework organizes existing maintenance records\n- **Current RPN baseline** ← Risk Priority Numbers for current maintenance approach\n\n**Analyze Phase Integration:**\n- **Root Cause Analysis** ← FMEA identifies failure causes systematically\n- **Fishbone Diagrams** ← FMEA provides structured input for cause categories\n- **Pareto Analysis** ← Prioritize failure modes by RPN (Risk Priority Number)\n- **Statistical analysis** ← Analyze failure patterns and maintenance effectiveness\n\n**Aviation FMEA Implementation:**\n\n**1. System Breakdown Structure:**\n- **Aircraft systems**: Engines, hydraulics, avionics, landing gear, etc.\n- **Maintenance processes**: Inspection, servicing, repair, testing\n- **Support systems**: Tools, facilities, personnel, documentation\n\n**2. Failure Mode Identification:**\n- **Component failures**: Bearing wear, seal degradation, electronic malfunctions\n- **Process failures**: Incomplete inspections, incorrect procedures, tool failures\n- **Human factors**: Maintenance errors, communication breakdowns, training gaps\n\n**3. Risk Assessment Matrix:**\n```\nRPN = Severity × Occurrence × Detection\n\nSeverity (1-10):\n- 1-3: Minor operational impact\n- 4-6: Moderate delays/costs\n- 7-8: Major safety/operational impact\n- 9-10: Catastrophic safety consequences\n\nOccurrence (1-10):\n- 1-2: Very rare (>1 year)\n- 3-4: Rare (6-12 months)\n- 5-6: Moderate (1-6 months)\n- 7-8: Frequent (weekly-monthly)\n- 9-10: Very frequent (daily)\n\nDetection (1-10):\n- 1-2: Almost certain detection\n- 3-4: High detection probability\n- 5-6: Moderate detection\n- 7-8: Low detection probability\n- 9-10: Very low/no detection\n```\n\n**Improve Phase Integration:**\n- **Preventive maintenance optimization** ← FMEA-driven maintenance intervals\n- **Condition monitoring** ← FMEA identifies critical parameters to monitor\n- **Maintenance procedure improvement** ← FMEA guides process standardization\n- **Training programs** ← FMEA identifies critical competency gaps\n\n**Control Phase Integration:**\n- **Control Plans** ← FMEA defines ongoing risk monitoring\n- **Key Performance Indicators** ← Track RPN reduction over time\n- **Standard Operating Procedures** ← Incorporate FMEA-based maintenance protocols\n- **Continuous monitoring** ← Update FMEA based on new failure data\n\n**Specific Aviation Applications:**\n\n**Engine Maintenance FMEA Example:**\n- **Failure Mode**: Turbine blade fatigue cracking\n- **Potential Effects**: Engine shutdown, emergency landing, passenger safety risk\n- **Potential Causes**: High cycle fatigue, foreign object damage, material defects\n- **Current Controls**: Scheduled borescope inspections every 500 hours\n- **Recommended Actions**: Implement vibration monitoring, reduce inspection intervals for high-cycle aircraft\n\n**Maintenance Process FMEA Example:**\n- **Failure Mode**: Incomplete hydraulic system inspection\n- **Potential Effects**: Undetected leaks, system failure during flight\n- **Potential Causes**: Time pressure, inadequate lighting, inspector fatigue\n- **Current Controls**: Checklist verification, supervisor sign-off\n- **Recommended Actions**: Improved inspection tools, mandatory rest periods, enhanced training\n\n**Integration with Other Six Sigma Tools:**\n\n**1. Design of Experiments (DOE):**\n- Use FMEA failure modes as experimental factors\n- Optimize maintenance intervals based on FMEA risk assessment\n- Test alternative maintenance procedures identified through FMEA\n\n**2. Statistical Process Control:**\n- Monitor failure rates for high-RPN failure modes\n- Track maintenance effectiveness metrics\n- Control charts for critical system reliability parameters\n\n**3. Lean Tools:**\n- **5S**: Organize maintenance areas to reduce FMEA-identified human error risks\n- **Standard Work**: Develop procedures based on FMEA critical control points\n- **Visual Management**: Display FMEA risk priorities in maintenance areas\n\n**Expected Business Benefits:**\n\n**Quantitative Improvements:**\n- **20-40% reduction** in unplanned maintenance events\n- **15-25% improvement** in aircraft availability\n- **10-30% reduction** in maintenance costs through optimized scheduling\n- **Significant reduction** in flight delays due to maintenance issues\n\n**Qualitative Benefits:**\n- **Enhanced safety culture** through systematic risk management\n- **Regulatory compliance** with proactive risk assessment documentation\n- **Improved maintenance planning** with data-driven prioritization\n- **Better resource allocation** focusing on highest-risk areas\n\n**Implementation Roadmap:**\n\n**Phase 1 (Months 1-2):** FMEA team training, system selection, initial FMEA development\n**Phase 2 (Months 3-4):** Risk assessment completion, action plan development\n**Phase 3 (Months 5-6):** Implementation of high-priority actions, control system setup\n**Phase 4 (Months 7-12):** Monitoring, continuous improvement, expansion to additional aircraft systems\n\nFMEA will transform your reactive maintenance approach into a proactive, risk-based system that improves safety, reduces costs, and enhances operational reliability.",
  "id": 305,
  "domain": "Transportation & Logistics",
  "sub_domain": "Aviation Maintenance"
}
```

## Content Guidelines

### Question Variety Requirements

**1. Explanation Questions (40% of samples):**
- "Tell me about...", "What is...", "Explain...", "How does... work?"
- Focus on LSS tools, statistical methods, concepts
- Provide educational, comprehensive explanations
- Include practical applications and industry examples

**2. Application/Suitability Questions (40% of samples):**
- "Can I use...", "Is... appropriate for...", "Would... work for...", "Should I..."
- Address specific business scenarios
- Provide decision frameworks and criteria
- Include pros/cons and implementation considerations

**3. Multi-Level Complex Questions (20% of samples):**
- Combine explanation with application
- Address multiple related concepts
- Provide comprehensive integration guidance
- Show tool relationships and DMAIC connections

### Content Quality Standards

**Educational Value:**
- Clear, step-by-step explanations
- Practical examples and applications
- Industry-specific contexts and use cases
- Integration with other LSS tools and methods

**Technical Accuracy:**
- Correct statistical concepts and procedures
- Appropriate tool selection criteria
- Valid assumption checking and alternatives
- Realistic business scenarios and outcomes

**Practical Focus:**
- Actionable implementation guidance
- Business impact and value proposition
- Common pitfalls and success factors
- Measurable outcomes and benefits

## Batch Management

### Batch Creation Process
- Create samples in batches of **8-12** samples
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

### Educational Effectiveness
- Clear concept explanations
- Practical application guidance
- Industry-relevant examples
- Integration with broader LSS methodology

### Importance of Quality
Diversity and high quality are critical because these samples will be used to fine-tune a Small Language Model (SLM), which will serve as a Master Black Belt AI Agent.

### Continuous Improvement
Monitor and adjust industry distribution in subsequent batches to maintain overall balance and achieve target percentages across the complete dataset.

## Project Context

### File Structure
- **Knowledge Base**: `knowledgebase/LSS_Methods.csv` and `knowledgebase/HypothesisTesting.csv`
- **Output Location**: `datasets/lss_CoT/lss_cot_batch{batch_number}.json`
- **Documentation**: Comprehensive tracking of industry distribution and quality metrics

### Technical Requirements
- **ID Management**: Sequential, unique integer IDs across all batches
- **Industry Balance**: Target percentages maintained within 1% variance
- **Quality Standards**: Expert-level educational content
- **Format Consistency**: Strict adherence to JSON structure

This project aims to create the highest quality FAQ-style educational dataset for developing an expert-level Master Black Belt AI Agent with comprehensive teaching and consultation capabilities.
