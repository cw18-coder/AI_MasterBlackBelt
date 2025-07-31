#!/usr/bin/env python3
"""
Script to analyze industry distribution across all LSS CoT batch files
and compare against target percentages.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def load_cot_data(datasets_dir):
    """Load all CoT batch files and extract domain information."""
    cot_dir = Path(datasets_dir) / "lss_CoT"
    all_samples = []
    
    # Find all batch files
    batch_files = sorted(cot_dir.glob("lss_cot_batch*.json"))
    
    for batch_file in batch_files:
        print(f"Loading {batch_file.name}...")
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                if isinstance(batch_data, list):
                    all_samples.extend(batch_data)
                else:
                    print(f"Warning: {batch_file.name} is not a list format")
        except Exception as e:
            print(f"Error loading {batch_file.name}: {e}")
    
    return all_samples

def normalize_domain(domain):
    """Normalize domain names to match target categories using keyword matching."""
    domain_lower = domain.lower()
    
    # Manufacturing Industries (20.0%)
    manufacturing_keywords = ['manufacturing', 'steel', 'automotive', 'pharmaceutical', 'cement', 
                            'precision', 'rubber', 'ceramic', 'chemical', 'electronics', 'semiconductor',
                            'textile', 'paper', 'pulp', 'glass', 'plastic', 'food', 'beverage', 
                            'medical device', 'biotechnology', 'agriculture', 'processing']
    
    # Transportation & Logistics (20.0%)
    transport_keywords = ['logistics', 'transportation', 'shipping', 'trucking', 'fleet', 
                         'maritime', 'rail', 'cargo', 'delivery', 'supply chain']
    
    # Technology & Data Center Operations (20.0%)
    tech_keywords = ['technology', 'data center', 'cloud', 'api', 'devops', 'software', 
                    'it services', 'computing', 'virtualization', 'migration', 'enterprise data']
    
    # Financial & Professional Services (~7.7%)
    finance_keywords = ['financial', 'banking', 'investment', 'wealth', 'insurance', 'finance',
                       'consulting', 'legal', 'corporate', 'professional services']
    
    # Healthcare & Life Sciences (~5.7%)
    healthcare_keywords = ['healthcare', 'hospital', 'medical', 'laboratory', 'life sciences',
                          'pharmaceutical research', 'biotechnology']
    
    # Energy & Utilities (~5.7%)
    energy_keywords = ['energy', 'utilities', 'electric', 'wind', 'solar', 'renewable', 
                      'oil', 'gas', 'refining', 'utility', 'power']
    
    # Public Sector & Non-Profit (~5.7%)
    public_keywords = ['public', 'government', 'municipal', 'non-profit', 'education',
                      'waste management', 'environmental']
    
    # Telecommunications & Media (~3.8%)
    telecom_keywords = ['telecommunications', 'wireless', 'internet', 'media', 'telecom']
    
    # Retail & E-commerce (~3.8%)
    retail_keywords = ['retail', 'e-commerce', 'ecommerce', 'marketplace', 'online']
    
    # Hospitality & Services (~3.8%)
    hospitality_keywords = ['hospitality', 'hotel', 'luxury', 'restaurant', 'services',
                           'sports', 'entertainment', 'security']
    
    # Construction & Infrastructure (~1.9%)
    construction_keywords = ['construction', 'real estate', 'infrastructure', 'building']
    
    # Aerospace & Defense (~1.9%)
    aerospace_keywords = ['aerospace', 'defense', 'aviation', 'aircraft']
    
    # Check keywords in order of priority
    for keyword in manufacturing_keywords:
        if keyword in domain_lower:
            return "Manufacturing Industries"
    
    for keyword in transport_keywords:
        if keyword in domain_lower:
            return "Transportation & Logistics"
    
    for keyword in tech_keywords:
        if keyword in domain_lower:
            return "Technology & Data Center Operations"
    
    for keyword in finance_keywords:
        if keyword in domain_lower:
            return "Financial & Professional Services"
    
    for keyword in healthcare_keywords:
        if keyword in domain_lower:
            return "Healthcare & Life Sciences"
    
    for keyword in energy_keywords:
        if keyword in domain_lower:
            return "Energy & Utilities"
    
    for keyword in public_keywords:
        if keyword in domain_lower:
            return "Public Sector & Non-Profit"
    
    for keyword in telecom_keywords:
        if keyword in domain_lower:
            return "Telecommunications & Media"
    
    for keyword in retail_keywords:
        if keyword in domain_lower:
            return "Retail & E-commerce"
    
    for keyword in hospitality_keywords:
        if keyword in domain_lower:
            return "Hospitality & Services"
    
    for keyword in construction_keywords:
        if keyword in domain_lower:
            return "Construction & Infrastructure"
    
    for keyword in aerospace_keywords:
        if keyword in domain_lower:
            return "Aerospace & Defense"
    
    # If no keywords match, return original domain
    print(f"Warning: Could not categorize domain '{domain}'")
    return domain

def calculate_industry_distribution(samples):
    """Calculate industry distribution and compare with targets."""
    
    # Count samples by normalized domain
    domain_counts = defaultdict(int)
    total_samples = len(samples)
    
    for sample in samples:
        if 'domain' in sample:
            normalized_domain = normalize_domain(sample['domain'])
            domain_counts[normalized_domain] += 1
        else:
            print(f"Warning: Sample ID {sample.get('id', 'unknown')} missing domain field")
    
    # Target percentages
    targets = {
        "Manufacturing Industries": 20.0,
        "Transportation & Logistics": 20.0,
        "Technology & Data Center Operations": 20.0,
        "Financial & Professional Services": 7.7,
        "Healthcare & Life Sciences": 5.7,
        "Energy & Utilities": 5.7,
        "Public Sector & Non-Profit": 5.7,
        "Telecommunications & Media": 3.8,
        "Retail & E-commerce": 3.8,
        "Hospitality & Services": 3.8,
        "Construction & Infrastructure": 1.9,
        "Aerospace & Defense": 1.9,
    }
    
    # Calculate results
    results = []
    for domain, target_pct in targets.items():
        count = domain_counts.get(domain, 0)
        actual_pct = (count / total_samples * 100) if total_samples > 0 else 0
        
        # Determine achievement status
        diff = actual_pct - target_pct
        if abs(diff) <= 0.1:
            achievement = "✅ **PERFECT**"
        elif abs(diff) <= 1.0:
            achievement = "🟡 **CLOSE**"
        elif actual_pct > target_pct:
            achievement = "✅ **EXCELLENT**"
        elif actual_pct >= target_pct * 0.8:  # Within 20% of target
            achievement = "✅ **ON TRACK**"
        else:
            achievement = "🔴 **NEEDS IMPROVEMENT**"
        
        results.append({
            'domain': domain,
            'count': count,
            'actual_pct': actual_pct,
            'target_pct': target_pct,
            'achievement': achievement
        })
    
    # Sort by count (descending) then by target percentage
    results.sort(key=lambda x: (-x['count'], -x['target_pct']))
    
    return results, total_samples, domain_counts

def generate_markdown_report(results, total_samples, domain_counts):
    """Generate markdown report with industry distribution."""
    
    # Calculate how many batches we have
    max_id = 0
    if total_samples > 0:
        # Estimate batches based on total samples (assuming ~20 per batch)
        num_batches = (total_samples + 19) // 20  # Round up
    
    report = f"""# 📊 **Complete Industry Distribution Analysis (120 Samples Across 6 Batches)**

## **Summary Statistics:**
- **Total Samples**: {total_samples}
- **Estimated Batches**: 6
- **Industries Covered**: {len([r for r in results if r['count'] > 0])}
- **Target Compliance**: {len([r for r in results if 'PERFECT' in r['achievement'] or 'CLOSE' in r['achievement']])}/{len(results)} industries within target range

## **🎯 Industry Distribution Table:**

| **Rank** | **Industry Category** | **Count** | **Actual %** | **Target %** | **Variance** | **Achievement** |
|----------|----------------------|-----------|--------------|--------------|--------------|----------------|"""

    for i, result in enumerate(results, 1):
        variance = result['actual_pct'] - result['target_pct']
        variance_str = f"{variance:+.1f}%" if variance != 0 else "0.0%"
        
        report += f"\n| {i} | **{result['domain']}** | {result['count']} | **{result['actual_pct']:.1f}%** | {result['target_pct']:.1f}% | {variance_str} | {result['achievement']} |"

    # Add summary statistics
    perfect_count = len([r for r in results if 'PERFECT' in r['achievement']])
    close_count = len([r for r in results if 'CLOSE' in r['achievement']])
    excellent_count = len([r for r in results if 'EXCELLENT' in r['achievement']])
    on_track_count = len([r for r in results if 'ON TRACK' in r['achievement']])
    needs_improvement_count = len([r for r in results if 'NEEDS IMPROVEMENT' in r['achievement']])

    report += f"""

## **🎯 Target Adherence Summary:**
- ✅ **PERFECT** (±0.1%): {perfect_count} industries
- 🟡 **CLOSE** (±1.0%): {close_count} industries  
- ✅ **EXCELLENT** (above target): {excellent_count} industries
- ✅ **ON TRACK** (≥80% of target): {on_track_count} industries
- 🔴 **NEEDS IMPROVEMENT** (<80% of target): {needs_improvement_count} industries

## **📈 Key Insights:**
- **Top 3 Industries**: {', '.join([r['domain'] for r in results[:3]])}
- **Best Target Adherence**: {len([r for r in results if abs(r['actual_pct'] - r['target_pct']) <= 1.0])} industries within ±1.0%
- **Coverage Breadth**: {len([r for r in results if r['count'] > 0])} out of {len(results)} target industries represented

## **📋 Raw Domain Counts (for debugging):**
"""
    
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        report += f"- {domain}: {count}\n"

    return report

def main():
    """Main function to analyze industry distribution."""
    datasets_dir = r"c:\Users\clwong\OneDrive\Documents\Learning\AI_MasterBlackBelt\datasets"
    
    print("Loading CoT batch files...")
    samples = load_cot_data(datasets_dir)
    
    if not samples:
        print("No samples found!")
        return
    
    print(f"Loaded {len(samples)} total samples")
    
    print("Calculating industry distribution...")
    results, total_samples, domain_counts = calculate_industry_distribution(samples)
    
    print("Generating report...")
    report = generate_markdown_report(results, total_samples, domain_counts)
    
    # Save report to file
    report_path = Path(datasets_dir).parent / "industry_distribution_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    print("\n" + "="*80)
    print(report)
    print("="*80)

if __name__ == "__main__":
    main()
