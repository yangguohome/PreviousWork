import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Set the base directory
a_values = np.arange(3, 12, 2)
b_values = [1, 3, 5, 7, 10]

# Initialize an empty list to collect all results for concatenation
all_results = []

for a in a_values:
    for b in b_values:
        currentName = f'MaxTree{a}_MinChild{b}'
        base_directory = Path(
            r"\\...\Yang_Guo\XGboost\results\XMass\MaxTree_MinChild") / currentName

        # Define file paths
        evidence_file = os.path.join(base_directory, 'evidence.txt')
        protein_file = os.path.join(base_directory, 'proteinGroups.txt')

        # Debugging print to check paths
        print(f"Checking path: {evidence_file}")
        if not evidence_file.exists():
            print(f"File not found: {evidence_file}")
            continue

        if not protein_file.exists():
            print(f"File not found: {protein_file}")
            continue

        # Load the evidence and protein data
        evidence_mixed = pd.read_csv(evidence_file, sep='\t')
        protein_mixed = pd.read_csv(protein_file, sep='\t')

        # Apply transformation to 'Taxonomy names' column in evidence
        evidence_mixed['Taxonomy names'] = evidence_mixed['Taxonomy names'].apply(
            lambda x: 'undecided' if isinstance(x, str) and ';' in x else ('unmapped' if pd.isna(x) else x)
        )

        # Calculate taxonomy counts
        taxonomy_counts = evidence_mixed['Taxonomy names'].value_counts()

        # Calculate specific ratio for Arabidopsis thaliana (FDR on peptides level)
        fdrPeptides = (
                taxonomy_counts.get('Arabidopsis thaliana', 0) /
                (taxonomy_counts.get('Arabidopsis thaliana', 0) +
                 taxonomy_counts.get('Saccharomyces cerevisiae', 0) +
                 taxonomy_counts.get('Escherichia coli', 0) +
                 taxonomy_counts.get('Homo sapiens', 0))
        )

        # Apply transformation to 'Species' column in protein_mixed
        protein_mixed['Species'] = protein_mixed['Species'].apply(
            lambda x: 'undecided' if isinstance(x, str) and ';' in x else ('unmapped' if pd.isna(x) else x)
        )

        # Calculate species counts
        pro_species_counts = protein_mixed['Species'].value_counts()

        # Calculate specific ratio for Arabidopsis thaliana (FDR on protein level)
        arabidopsis_protein_ratio = (
                pro_species_counts.get('Arabidopsis thaliana', 0) /
                (pro_species_counts.get('Arabidopsis thaliana', 0) +
                 pro_species_counts.get('Saccharomyces cerevisiae', 0) +
                 pro_species_counts.get('Escherichia coli', 0) +
                 pro_species_counts.get('Homo sapiens', 0))
        )

        # Create a summary DataFrame for the current combination
        summary = pd.DataFrame({
            "Parameter Combination": [currentName],
            "FDR Peptides": [fdrPeptides],
            "FDR Proteins": [arabidopsis_protein_ratio],
            "Protein Counts Human": [pro_species_counts.get('Homo sapiens', 0)],
            "Protein Counts Arabidopsis": [pro_species_counts.get('Arabidopsis thaliana', 0)]
        })

        # Append the summary DataFrame to the list
        all_results.append(summary)


# Sort for consistency
df = all_results.sort_values("Parameter Combination").copy()
x = np.arange(len(df))
labels = df["Parameter Combination"]

# calculate total counts
arab_pep = df["Peptide Counts Arabidopsis"]
human_pep = df["Peptide Counts Human"]
total_pep = arab_pep + human_pep

# Plot stacked bars
plt.figure(figsize=(16, 6))
bar1 = plt.bar(x, human_pep, label="Peptide Counts Human", color='steelblue')
bar2 = plt.bar(x, arab_pep, bottom=human_pep, label="Peptide Counts Arabidopsis", color='lightblue')

# Add FDR % label on top of each bar
for i in range(len(x)):
    total = human_pep.iloc[i] + arab_pep.iloc[i]
    fdr = arab_pep.iloc[i] / total if total != 0 else 0
    fdr_text = f"{fdr:.2%}"
    plt.text(x[i], total + total * 0.01, fdr_text, ha='center', va='bottom', fontsize=8, color='black')

# Final touch
plt.xticks(x, labels, rotation=90)
plt.ylabel("Peptide Counts")
plt.title("XMass_Orbitrap_MaxTree_MinChild Peptides")
plt.legend()
plt.tight_layout()
plt.savefig(r"\\...\Yang_Guo\XGboost\results\XMass\Final_Stacked_Peptide_FDR_Label.png")
plt.show()

