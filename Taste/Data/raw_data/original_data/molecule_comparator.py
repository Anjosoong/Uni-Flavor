#!/usr/bin/env python3
"""
Molecular SMILES Comparison
"""

import os
import sys
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from rdkit import Chem
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# File paths - modify these for direct execution
file1_path = "Tastants_final_complete_5.csv"
file2_path = "Full_dataset_fourtaste.csv"

@dataclass
class ComparisonConfig:
    skip_invalid_smiles: bool = True
    remove_stereochemistry: bool = False  # Keep stereochemistry
    canonicalize: bool = True

@dataclass
class ComparisonResult:
    common_molecules: pd.DataFrame
    file1_unique: pd.DataFrame
    file2_unique: pd.DataFrame
    statistics: Dict[str, int]

class FileReader:
    def read_excel(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_excel(file_path)
    
    def read_csv(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        encodings = ['utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read file: {file_path}")
    
    def extract_smiles_column(self, df: pd.DataFrame) -> str:
        possible_names = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles']
        
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                return col
        
        # Check first column
        if len(df.columns) > 0:
            first_col = df.columns[0]
            sample_values = df[first_col].dropna().head(10)
            smiles_like_count = sum(1 for value in sample_values 
                                  if isinstance(value, str) and any(char in value for char in ['C', 'N', 'O', '(', ')']))
            if smiles_like_count >= len(sample_values) * 0.5:
                return first_col
        
        raise ValueError("Cannot find SMILES column")

class MoleculeProcessor:
    def __init__(self, config: ComparisonConfig):
        self.config = config
    
    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """
        Standardize SMILES using RDKit canonical form
        - Preserves stereochemistry (chiral centers and E/Z bonds)
        - Ensures consistent atom ordering
        - Handles tautomers and resonance forms
        """
        if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == '':
            return None
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None
            
            # Remove stereochemistry if configured
            if self.config.remove_stereochemistry:
                Chem.RemoveStereochemistry(mol)
            
            # Generate canonical SMILES with stereochemistry preserved
            # isomericSmiles=True ensures stereochemistry is included in output
            canonical_smiles = Chem.MolToSmiles(
                mol, 
                canonical=self.config.canonicalize,
                isomericSmiles=not self.config.remove_stereochemistry
            )
            
            return canonical_smiles
        except:
            return None
    
    def process_dataframe(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        processed_df = df.copy()
        processed_df['standardized_smiles'] = None
        processed_df['is_valid'] = False
        
        for idx, row in processed_df.iterrows():
            original_smiles = row[smiles_col]
            standardized = self.standardize_smiles(original_smiles)
            if standardized:
                processed_df.at[idx, 'standardized_smiles'] = standardized
                processed_df.at[idx, 'is_valid'] = True
        
        if self.config.skip_invalid_smiles:
            processed_df = processed_df[processed_df['is_valid']].copy()
        
        return processed_df

class ComparisonAnalyzer:
    def find_common_molecules(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        smiles1 = set(df1['standardized_smiles'].dropna())
        smiles2 = set(df2['standardized_smiles'].dropna())
        common_smiles = smiles1.intersection(smiles2)
        
        if len(common_smiles) == 0:
            return pd.DataFrame()
        
        common_df1 = df1[df1['standardized_smiles'].isin(common_smiles)].copy()
        common_df2 = df2[df2['standardized_smiles'].isin(common_smiles)].copy()
        
        merged_data = []
        for smiles in common_smiles:
            data1 = common_df1[common_df1['standardized_smiles'] == smiles].iloc[0].to_dict()
            data2 = common_df2[common_df2['standardized_smiles'] == smiles].iloc[0].to_dict()
            
            merged_row = data1.copy()
            for key, value in data2.items():
                if key not in merged_row or pd.isna(merged_row[key]):
                    merged_row[key] = value
                elif key != 'standardized_smiles' and key != 'is_valid':
                    if not pd.isna(value) and value != merged_row[key]:
                        merged_row[f'file2_{key}'] = value
            
            merged_row['source_file'] = 'both'
            merged_data.append(merged_row)
        
        return pd.DataFrame(merged_data)
    
    def find_unique_molecules(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        smiles1 = set(df1['standardized_smiles'].dropna())
        smiles2 = set(df2['standardized_smiles'].dropna())
        
        unique_smiles1 = smiles1 - smiles2
        unique_smiles2 = smiles2 - smiles1
        
        unique_df1 = df1[df1['standardized_smiles'].isin(unique_smiles1)].copy()
        unique_df2 = df2[df2['standardized_smiles'].isin(unique_smiles2)].copy()
        
        unique_df1['source_file'] = 'file1'
        unique_df2['source_file'] = 'file2'
        
        return unique_df1, unique_df2
    
    def merge_duplicate_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.drop_duplicates(subset=['standardized_smiles'], keep='first')
    
    def get_comparison_stats(self, common_df: pd.DataFrame, unique_df1: pd.DataFrame, 
                           unique_df2: pd.DataFrame) -> Dict[str, int]:
        return {
            'common_molecules': len(common_df.drop_duplicates(subset=['standardized_smiles'])) if not common_df.empty else 0,
            'file1_unique_molecules': len(unique_df1),
            'file2_unique_molecules': len(unique_df2),
            'total_unique_molecules': len(common_df.drop_duplicates(subset=['standardized_smiles'])) + len(unique_df1) + len(unique_df2) if not common_df.empty else len(unique_df1) + len(unique_df2)
        }

class ResultWriter:
    def write_results(self, result: ComparisonResult, output_dir: str = ".", file1_name: str = "file1", file2_name: str = "file2", file1_annotated: pd.DataFrame = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # file1 annotated result (original data with comparison_status column)
        file1_annotated_path = os.path.join(output_dir, f"{file1_name}_annotated.csv")
        # file2 unique molecules
        unique2_path = os.path.join(output_dir, f"{file2_name}_unique.csv")
        
        if file1_annotated is not None and not file1_annotated.empty:
            file1_annotated.to_csv(file1_annotated_path, index=False, encoding='utf-8-sig')
        if not result.file2_unique.empty:
            result.file2_unique.to_csv(unique2_path, index=False, encoding='utf-8-sig')

class MoleculeComparator:
    def __init__(self, file1_path: str, file2_path: str, config: ComparisonConfig = None):
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.config = config or ComparisonConfig()
        
        self.file_reader = FileReader()
        self.molecule_processor = MoleculeProcessor(self.config)
        self.comparison_analyzer = ComparisonAnalyzer()
        self.result_writer = ResultWriter()
    
    def run_comparison(self) -> Tuple[ComparisonResult, pd.DataFrame]:
        print("Processing...")
        print("Step 1: Reading files...")
        
        # Read files
        if self.file1_path.endswith(('.xlsx', '.xls')):
            df1 = self.file_reader.read_excel(self.file1_path)
        else:
            df1 = self.file_reader.read_csv(self.file1_path)
        
        if self.file2_path.endswith('.csv'):
            df2 = self.file_reader.read_csv(self.file2_path)
        else:
            df2 = self.file_reader.read_excel(self.file2_path)
        
        print(f"  File1: {len(df1)} molecules")
        print(f"  File2: {len(df2)} molecules")
        
        # Detect SMILES columns
        smiles_col1 = self.file_reader.extract_smiles_column(df1)
        smiles_col2 = self.file_reader.extract_smiles_column(df2)
        
        print(f"\nStep 2: Standardizing SMILES...")
        print(f"  Method: Canonical SMILES with stereochemistry preserved")
        
        # Process molecular data
        processed_df1 = self.molecule_processor.process_dataframe(df1, smiles_col1)
        processed_df2 = self.molecule_processor.process_dataframe(df2, smiles_col2)
        
        valid1 = len(processed_df1)
        valid2 = len(processed_df2)
        print(f"  File1: {valid1} valid molecules ({len(df1)-valid1} failed)")
        print(f"  File2: {valid2} valid molecules ({len(df2)-valid2} failed)")
        
        print(f"\nStep 3: Comparing molecules...")
        
        # Get SMILES sets for comparison
        smiles1 = set(processed_df1['standardized_smiles'].dropna())
        smiles2 = set(processed_df2['standardized_smiles'].dropna())
        common_smiles = smiles1.intersection(smiles2)
        
        print(f"  Unique in File1: {len(smiles1)}")
        print(f"  Unique in File2: {len(smiles2)}")
        print(f"  Common: {len(common_smiles)}")
        
        # Create file2 lookup dict by standardized_smiles
        file2_lookup = {}
        for _, row in processed_df2.iterrows():
            smiles = row['standardized_smiles']
            if pd.notna(smiles):
                file2_lookup[smiles] = row.to_dict()
        
        print(f"\nStep 4: Annotating results...")
        
        # Annotate file1 with comparison status and merge file2 info for common molecules
        file1_annotated = processed_df1.copy()
        file1_annotated['comparison_status'] = 'unique'
        
        # Get file2 columns (exclude internal columns)
        file2_cols = [c for c in processed_df2.columns if c not in ['standardized_smiles', 'is_valid']]
        
        # Add file2 columns with prefix
        for col in file2_cols:
            file1_annotated[f'file2_{col}'] = None
        
        # Merge file2 info for common molecules
        for idx, row in file1_annotated.iterrows():
            smiles = row['standardized_smiles']
            if smiles in common_smiles:
                file1_annotated.at[idx, 'comparison_status'] = 'common'
                # Add file2 info
                if smiles in file2_lookup:
                    file2_data = file2_lookup[smiles]
                    for col in file2_cols:
                        if col in file2_data:
                            file1_annotated.at[idx, f'file2_{col}'] = file2_data[col]
        
        # Find file2 unique molecules
        unique_smiles2 = smiles2 - smiles1
        unique_df2 = processed_df2[processed_df2['standardized_smiles'].isin(unique_smiles2)].copy()
        unique_df2 = unique_df2.drop_duplicates(subset=['standardized_smiles'], keep='first')
        
        # Generate statistics
        statistics = {
            'file1_total': len(file1_annotated),
            'common_molecules': len(file1_annotated[file1_annotated['comparison_status'] == 'common']),
            'file1_unique_molecules': len(file1_annotated[file1_annotated['comparison_status'] == 'unique']),
            'file2_unique_molecules': len(unique_df2)
        }
        
        result = ComparisonResult(
            common_molecules=pd.DataFrame(),
            file1_unique=pd.DataFrame(),
            file2_unique=unique_df2,
            statistics=statistics
        )
        
        return result, file1_annotated

def main():
    # Check if command line arguments provided
    if len(sys.argv) >= 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        # Use default file paths defined at top of script
        file1 = file1_path
        file2 = file2_path
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"Error: File not found {file1}")
        sys.exit(1)
    
    if not os.path.exists(file2):
        print(f"Error: File not found {file2}")
        sys.exit(1)
    
    try:
        # Create comparator and run
        comparator = MoleculeComparator(file1, file2)
        result, file1_annotated = comparator.run_comparison()
        
        # Write results
        file1_name = os.path.splitext(os.path.basename(file1))[0]
        file2_name = os.path.splitext(os.path.basename(file2))[0]
        comparator.result_writer.write_results(result, ".", file1_name, file2_name, file1_annotated)
        
        # Show results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"Standardization: Canonical SMILES (stereochemistry preserved)")
        print(f"\nFile1 ({file1_name}):")
        print(f"  Total: {result.statistics.get('file1_total', 0)}")
        print(f"  Common with File2: {result.statistics.get('common_molecules', 0)}")
        print(f"  Unique to File1: {result.statistics.get('file1_unique_molecules', 0)}")
        print(f"\nFile2 ({file2_name}):")
        print(f"  Unique to File2: {result.statistics.get('file2_unique_molecules', 0)}")
        print(f"\nOutput files:")
        print(f"  {file1_name}_annotated.csv")
        print(f"  {file2_name}_unique.csv")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Usage:
1. Modify file1_path and file2_path at top of script, then run: python molecule_comparator.py
2. Or use command line: python molecule_comparator.py file1.xlsx file2.csv

Output files: 
- [file1]_annotated.csv: file1 original data with comparison_status column (common/unique)
- [file2]_unique.csv: molecules only in file2
"""