"""
Feature extraction utilities for peptide sequences.

Provides functions to extract biophysical, compositional, positional,
and dipeptide features from amino acid sequences for use in ML models.
"""

import itertools
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from typing import Dict, List, Optional


# =============================================================================
# Amino Acid Property Scales
# =============================================================================

# Kyte-Doolittle hydrophobicity scale
KD_HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Van der Waals volume
VDW_VOLUME = {
    'A': 67, 'C': 86, 'D': 91, 'E': 109, 'F': 135,
    'G': 48, 'H': 118, 'I': 124, 'K': 135, 'L': 124,
    'M': 124, 'N': 96, 'P': 90, 'Q': 114, 'R': 148,
    'S': 73, 'T': 93, 'V': 105, 'W': 163, 'Y': 141
}

VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
HYDROPHOBIC_AAS = 'AVLIMFPWC'
CHARGED_AAS = 'DEKRH'
POLAR_AAS = 'STNQ'
POSITIVE_AAS = 'RK'
NEGATIVE_AAS = 'DE'


# =============================================================================
# Core Feature Extraction
# =============================================================================

def validate_sequence(sequence: str) -> str:
    """Validate and clean a peptide sequence.

    Args:
        sequence: Amino acid sequence string.

    Returns:
        Cleaned uppercase sequence with ambiguous residues removed.

    Raises:
        ValueError: If sequence is empty or contains invalid characters.
    """
    if not sequence or not isinstance(sequence, str):
        raise ValueError("Sequence must be a non-empty string")
    sequence = sequence.upper().replace('X', '')
    if not all(aa in VALID_AMINO_ACIDS for aa in sequence):
        invalid = set(sequence) - VALID_AMINO_ACIDS
        raise ValueError(f"Sequence contains invalid amino acids: {invalid}")
    return sequence


def extract_biophysical_features(sequence: str) -> Dict[str, float]:
    """Extract biophysical properties from a peptide sequence.

    Uses BioPython's ProteinAnalysis to compute properties like
    hydrophobicity (GRAVY), isoelectric point, molecular weight,
    secondary structure fractions, and flexibility.

    Args:
        sequence: Validated amino acid sequence.

    Returns:
        Dictionary of biophysical feature names to values.
    """
    protein = ProteinAnalysis(sequence)
    sec_struct = protein.secondary_structure_fraction()
    flexibility = protein.flexibility()

    return {
        'seq_length': len(sequence),
        'gravy': protein.gravy(),
        'isoelectric_point': protein.isoelectric_point(),
        'aromaticity': protein.aromaticity(),
        'instability_index': protein.instability_index(),
        'mol_weight': protein.molecular_weight(),
        'helix_frac': sec_struct[0],
        'turn_frac': sec_struct[1],
        'sheet_frac': sec_struct[2],
        'flex_mean': np.mean(flexibility) if len(flexibility) > 0 else 0.0,
        'flex_std': np.std(flexibility) if len(flexibility) > 0 else 0.0,
    }


def extract_composition_features(sequence: str) -> Dict[str, float]:
    """Extract amino acid composition and group ratio features.

    Args:
        sequence: Validated amino acid sequence.

    Returns:
        Dictionary of composition feature names to values.
    """
    n = len(sequence)
    protein = ProteinAnalysis(sequence)
    aa_percent = protein.get_amino_acids_percent()

    features = {}

    # Individual amino acid percentages
    for aa in VALID_AMINO_ACIDS:
        features[f'aa_percent_{aa}'] = aa_percent.get(aa, 0.0)

    # Group ratios
    features['hydrophobic_ratio'] = sum(sequence.count(aa) for aa in HYDROPHOBIC_AAS) / n
    features['charged_ratio'] = sum(sequence.count(aa) for aa in CHARGED_AAS) / n
    features['polar_ratio'] = sum(sequence.count(aa) for aa in POLAR_AAS) / n
    features['aromatic_ratio'] = sum(sequence.count(aa) for aa in 'FYW') / n

    # Charge ratios
    features['positive_charge_ratio'] = sum(sequence.count(aa) for aa in POSITIVE_AAS) / n
    features['negative_charge_ratio'] = sum(sequence.count(aa) for aa in NEGATIVE_AAS) / n
    features['net_charge'] = (
        sum(sequence.count(aa) for aa in POSITIVE_AAS)
        - sum(sequence.count(aa) for aa in NEGATIVE_AAS)
    )

    return features


def extract_positional_features(sequence: str) -> Dict[str, float]:
    """Extract position-specific features (N-terminal and C-terminal).

    Args:
        sequence: Validated amino acid sequence.

    Returns:
        Dictionary of positional feature names to values.
    """
    n_term = sequence[:3]
    c_term = sequence[-3:]
    polar_set = set('STNQYCW')

    return {
        'n_term_charge': (
            sum(1 for aa in n_term if aa in POSITIVE_AAS)
            - sum(1 for aa in n_term if aa in NEGATIVE_AAS)
        ),
        'c_term_charge': (
            sum(1 for aa in c_term if aa in POSITIVE_AAS)
            - sum(1 for aa in c_term if aa in NEGATIVE_AAS)
        ),
        'n_term_hydrophobic': sum(1 for aa in n_term if aa in HYDROPHOBIC_AAS) / 3,
        'c_term_hydrophobic': sum(1 for aa in c_term if aa in HYDROPHOBIC_AAS) / 3,
        'n_term_polar': 1 if sequence[0] in polar_set else 0,
    }


def extract_physicochemical_features(sequence: str) -> Dict[str, float]:
    """Extract physicochemical property scale features.

    Computes mean Kyte-Doolittle hydrophobicity and mean
    Van der Waals volume across the sequence.

    Args:
        sequence: Validated amino acid sequence.

    Returns:
        Dictionary of physicochemical feature names to values.
    """
    n = len(sequence)
    return {
        'kd_hydrophobicity': sum(KD_HYDROPHOBICITY[aa] for aa in sequence) / n,
        'avg_vdw_volume': sum(VDW_VOLUME[aa] for aa in sequence) / n,
    }


def extract_dipeptide_features(sequence: str) -> Dict[str, float]:
    """Extract normalized dipeptide frequency features (400 features).

    Counts all 20x20 = 400 possible consecutive amino acid pairs
    and normalizes by total number of dipeptides in the sequence.

    Args:
        sequence: Validated amino acid sequence.

    Returns:
        Dictionary mapping dipeptide names to normalized frequencies.
    """
    aa_list = sorted(VALID_AMINO_ACIDS)
    dipeptides = [''.join(dp) for dp in itertools.product(aa_list, repeat=2)]
    total = max(len(sequence) - 1, 1)

    counts = {dp: 0 for dp in dipeptides}
    for i in range(len(sequence) - 1):
        dp = sequence[i:i + 2]
        if dp in counts:
            counts[dp] += 1

    return {f'dipep_{dp}': count / total for dp, count in counts.items()}


def extract_all_features(
    sequence: str,
    include_dipeptides: bool = False,
    additional_data: Optional[Dict] = None,
) -> Optional[Dict[str, float]]:
    """Extract the full feature set from a peptide sequence.

    Combines biophysical, composition, positional, and physicochemical
    features. Optionally includes dipeptide frequencies and additional
    dataset-provided features.

    Args:
        sequence: Raw amino acid sequence string.
        include_dipeptides: Whether to include 400 dipeptide features.
        additional_data: Optional dict of extra features from the dataset
            (e.g., 'aliphatic index', 'charge', 'hydrophobicity',
            'instability index', 'Isoelectric point', 'Molecular weight',
            'Structural class').

    Returns:
        Dictionary of all extracted features, or None if extraction fails.
    """
    try:
        seq = validate_sequence(sequence)
        features = {}
        features.update(extract_biophysical_features(seq))
        features.update(extract_composition_features(seq))
        features.update(extract_positional_features(seq))
        features.update(extract_physicochemical_features(seq))

        if include_dipeptides:
            features.update(extract_dipeptide_features(seq))

        # Incorporate additional dataset-provided features
        if additional_data:
            for col in ['aliphatic index', 'charge', 'hydrophobicity',
                        'instability index', 'Isoelectric point', 'Molecular weight']:
                if col in additional_data and additional_data[col] is not None:
                    key = 'dataset_' + col.lower().replace(' ', '_').replace('.', '')
                    features[key] = float(additional_data[col])

            if 'Structural class' in additional_data and additional_data['Structural class'] is not None:
                struct = str(additional_data['Structural class']).lower()
                for cls in ['alpha', 'beta', 'alpha+beta', 'coil', 'unknown']:
                    features[f'struct_class_{cls}'] = 1.0 if struct == cls else 0.0

        return features

    except Exception as e:
        print(f"Error processing sequence '{sequence}': {e}")
        return None


def extract_features_dataframe(
    sequences: List[str],
    include_dipeptides: bool = False,
    additional_data_list: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Extract features for a list of sequences and return as a DataFrame.

    Args:
        sequences: List of peptide sequences.
        include_dipeptides: Whether to include dipeptide features.
        additional_data_list: Optional list of dicts with extra features,
            one per sequence. Must match the length of sequences.

    Returns:
        DataFrame where each row corresponds to a sequence.
    """
    features_list = []
    for i, seq in enumerate(sequences):
        extra = additional_data_list[i] if additional_data_list else None
        feats = extract_all_features(seq, include_dipeptides, extra)
        if feats is not None:
            features_list.append(feats)
        else:
            features_list.append({})
    return pd.DataFrame(features_list)
