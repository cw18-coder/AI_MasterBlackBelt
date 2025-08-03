# GitHub Copilot Instruction Switching Guide

## How to Specify Instruction Files for Sample Generation

### Method 1: Direct File Reference
You can explicitly mention the instruction file you want me to use:

```
"Use the copilot-dmaic.instructions.md file to create a batch of 10 samples"
"Follow the copilot-hypothesis.instructions.md guidelines to generate samples"
"Create samples using copilot-faq.instructions.md format"
"Generate data reasoning samples per copilot-datareasoning.instructions.md"
```

### Method 2: Sample Type Keywords
You can use descriptive keywords that map to specific instruction files:

**For DMAIC samples:**
- "Create DMAIC samples"
- "Generate Define-Measure-Analyze-Improve-Control examples"
- "Make samples for DMAIC methodology training"

**For Hypothesis Testing samples:**
- "Create hypothesis testing samples"
- "Generate statistical test selection examples"
- "Make samples for choosing appropriate tests"

**For FAQ samples:**
- "Create FAQ-style samples"
- "Generate educational explanation samples"
- "Make samples that explain LSS concepts"

**For Data Reasoning samples:**
- "Create data reasoning samples"
- "Generate method recommendation examples"
- "Make samples for statistical analysis guidance"

### Method 3: Context-Based Selection
You can describe the learning objective, and I'll choose the appropriate instruction file:

```
"I want samples that teach students how to work through DMAIC projects step-by-step"
→ I'll use copilot-dmaic.instructions.md

"I need samples that help people choose the right statistical test for their data"
→ I'll use copilot-hypothesis.instructions.md

"Create samples that answer common questions about LSS tools and methods"
→ I'll use copilot-faq.instructions.md

"I want samples that show how to analyze complex datasets and recommend methods"
→ I'll use copilot-datareasoning.instructions.md
```

## Alignment with GitHub Copilot Agent Mode Best Practices

### ✅ **Follows Best Practices:**

**1. Context-Aware Instructions:**
- Each instruction file provides specific context for different sample types
- Clear role definition (Master Black Belt expertise)
- Detailed format specifications and examples
- Industry-specific guidance and constraints

**2. Modular Instruction Architecture:**
- Separate files for distinct capabilities (DMAIC, hypothesis testing, FAQ, data reasoning)
- Each file is self-contained with complete guidance
- Consistent structure across all instruction files
- Easy to maintain and update individually

**3. Clear Task Boundaries:**
- Each instruction file defines a specific domain of expertise
- Non-overlapping responsibilities between files
- Clear success metrics and quality standards
- Specific output formats and requirements

**4. Scalable Design:**
- Easy to add new instruction files for additional sample types
- Consistent naming convention: `copilot-{type}.instructions.md`
- Reusable patterns across instruction files
- Standardized batch management and ID tracking

### ✅ **Advanced Agent Mode Features:**

**1. Dynamic Context Selection:**
- I can automatically select the appropriate instruction file based on your request
- Context switching between different expertise modes
- Maintains consistency within each mode while adapting between them

**2. Knowledge Base Integration:**
- All instruction files reference the same knowledge base files (`LSS_Methods.csv`, `HypothesisTesting.csv`)
- Consistent industry distribution tracking across all sample types
- Unified ID management system across all batches

**3. Quality Assurance:**
- Each instruction file includes specific quality metrics
- Consistent validation criteria across sample types
- Built-in industry distribution tracking and balance requirements

## Recommended Usage Patterns

### For Training Dataset Creation:
```
"Create a batch of 10 DMAIC samples" → copilot-dmaic.instructions.md
"Generate 8 hypothesis testing samples" → copilot-hypothesis.instructions.md
"Make 12 FAQ samples covering various LSS topics" → copilot-faq.instructions.md
"Create 10 data reasoning samples" → copilot-datareasoning.instructions.md
```

### For Mixed Batches:
```
"Create a mixed batch with 3 DMAIC, 3 hypothesis testing, 3 FAQ, and 3 data reasoning samples"
→ I'll use all four instruction files appropriately
```

### For Specific Learning Objectives:
```
"I need samples that teach statistical test selection" → copilot-hypothesis.instructions.md
"Create samples for explaining LSS concepts to beginners" → copilot-faq.instructions.md
"Generate samples for complex data analysis scenarios" → copilot-datareasoning.instructions.md
```

## GitHub Copilot Agent Mode Advantages

**1. **Instruction Persistence:** Each file remains loaded and available for context switching
**2. **Role Specialization:** I can adopt different expert personas based on the instruction file
**3. **Quality Consistency:** Standardized quality metrics across all sample types
**4. **Scalable Architecture:** Easy to add new instruction files for additional capabilities
**5. **Context Awareness:** I understand the relationships between different sample types and can recommend appropriate combinations

This approach aligns perfectly with GitHub Copilot's Agent Mode philosophy of providing specialized, context-aware assistance while maintaining consistency and quality across different domains of expertise.
