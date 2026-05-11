#!/usr/bin/env python3
"""
Molecular Descriptor Calculation Pipeline using Mordred

Batch processing pipeline for calculating molecular descriptors from SMILES strings.
Memory-efficient implementation suitable for large molecular datasets.

Dependencies: pandas, mordred, rdkit, tqdm
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm


@dataclass
class DescriptorConfig:
    """Configuration for molecular descriptor calculation pipeline."""
    
    input_path: str = r"./Data/processed_data/odor_data/test_data.csv"
    output_path: str = r"./mordred/mordred_descriptors/test_descriptors.csv"
    batch_size: int = 32
    encoding: str = "utf-8"
    log_level: int = logging.INFO
    ignore_3d: bool = True
    required_columns: Tuple[str, ...] = ("SMILES",)
    smiles_column: str = "SMILES"
    name_column: str = "SMILES"  # Can be different from smiles_column for ID tracking
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if not self.input_path:
            raise ValueError("Input path cannot be empty")
        if not self.output_path:
            raise ValueError("Output path cannot be empty")
        
        # Automatically include name_column in required_columns if different from smiles_column
        if self.name_column and self.name_column not in self.required_columns:
            self.required_columns = tuple(dict.fromkeys((*self.required_columns, self.name_column)))


def configure_logging(log_level: int = logging.INFO) -> None:
    """Configure logging with consistent format."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def create_descriptor_calculator(ignore_3d: bool = True) -> Calculator:
    """Create Mordred calculator for 2D molecular descriptors."""
    return Calculator(descriptors, ignore_3D=ignore_3d)


def load_smiles_dataframe(csv_path: str, encoding: str) -> "pd.DataFrame":
    """Load SMILES dataset from CSV file with encoding fallback."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    
    # Try multiple encodings in order of preference (avoid duplicates)
    encodings_to_try = [encoding]
    fallback_encodings = ["utf-8-sig", "utf-8", "gbk", "cp1252"]
    for enc in fallback_encodings:
        if enc not in encodings_to_try:
            encodings_to_try.append(enc)
    
    for enc in encodings_to_try:
        try:
            dataframe = pd.read_csv(csv_path, encoding=enc)
            if enc != encoding:
                logging.info("Successfully loaded with encoding '%s' (fallback from '%s')", enc, encoding)
            else:
                logging.info("Successfully loaded with encoding '%s'", enc)
            logging.info("Loaded %d rows from %s", len(dataframe), csv_path)
            return dataframe
        except UnicodeDecodeError:
            if enc == encodings_to_try[-1]:  # Last encoding failed
                raise ValueError(f"Could not decode file with any of these encodings: {encodings_to_try}")
            continue
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
    
    raise ValueError(f"Could not load file with any encoding: {encodings_to_try}")


def validate_smiles_dataframe(dataframe: "pd.DataFrame", required_columns: Iterable[str]) -> None:
    """Validate DataFrame contains required columns."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        missing_column_names = ", ".join(missing_columns)
        raise ValueError(f"DataFrame must contain columns: {missing_column_names}")


def convert_smiles_to_molecules(
    smiles: Sequence[str],
    names: Sequence[str],
) -> Tuple[List[Mol], List[str], List[str], List[Tuple[str, str]]]:
    """Convert SMILES strings to RDKit molecule objects, filtering invalid entries.
    
    Returns:
        Tuple of (valid_molecules, valid_smiles, valid_names, invalid_entries)
    """
    valid_molecules: List[Mol] = []
    valid_smiles: List[str] = []
    valid_names: List[str] = []
    invalid_entries: List[Tuple[str, str]] = []

    for smiles_string, name in zip(smiles, names):
        # Handle missing values and non-string types (more comprehensive than math.isnan)
        if pd.isna(smiles_string):
            invalid_entries.append((name, str(smiles_string)))
            continue
        
        smiles_string = str(smiles_string).strip()
        if not smiles_string:
            invalid_entries.append((name, smiles_string))
            continue
        
        try:
            molecule = Chem.MolFromSmiles(smiles_string)
            if molecule is None:
                invalid_entries.append((name, smiles_string))
                continue
            valid_molecules.append(molecule)
            valid_smiles.append(smiles_string)  # Keep original SMILES
            valid_names.append(name)
        except Exception as e:
            logging.debug("Failed to parse SMILES '%s' for %s: %s", smiles_string, name, e)
            invalid_entries.append((name, smiles_string))

    if invalid_entries:
        sample_entries = ", ".join(
            f"{entry_name}:{entry_smiles}" for entry_name, entry_smiles in invalid_entries[:5]
        )
        logging.warning(
            "Skipped %d invalid SMILES entries (%.1f%%). Examples: %s",
            len(invalid_entries),
            100 * len(invalid_entries) / len(smiles) if smiles else 0,
            sample_entries,
        )

    return valid_molecules, valid_smiles, valid_names, invalid_entries


def generate_batches(
    molecules: Sequence[Mol],
    smiles: Sequence[str],
    names: Sequence[str],
    batch_size: int,
) -> Iterator[Tuple[Sequence[Mol], Sequence[str], Sequence[str]]]:
    """Generate fixed-size batches of molecules for memory-efficient processing."""
    for start_index in range(0, len(molecules), batch_size):
        end_index = start_index + batch_size
        yield (
            molecules[start_index:end_index], 
            smiles[start_index:end_index], 
            names[start_index:end_index]
        )


def clean_descriptor_data(dataframe: "pd.DataFrame", skip_columns: Sequence[str]) -> "pd.DataFrame":
    """Clean descriptor data by handling inf values and converting descriptor columns to numeric."""
    # Replace inf/-inf with NaN (using float("nan") for better compatibility)
    dataframe = dataframe.replace([float("inf"), float("-inf")], float("nan"))
    
    # Convert descriptor columns to numeric (coerce errors to NaN) - only process object columns for performance
    descriptor_cols = [c for c in dataframe.columns if c not in set(skip_columns)]
    obj_cols = [c for c in descriptor_cols if dataframe[c].dtype == "object"]
    if obj_cols:
        dataframe[obj_cols] = dataframe[obj_cols].apply(pd.to_numeric, errors="coerce")
    
    # Log columns with high NaN percentage
    nan_percentages = dataframe.isnull().mean() * 100
    high_nan_cols = nan_percentages[nan_percentages > 50]
    if not high_nan_cols.empty:
        logging.debug("Columns with >50%% NaN values: %s", high_nan_cols.to_dict())
    
    return dataframe


def calculate_descriptor_batches(
    calculator: Calculator,
    molecules: Sequence[Mol],
    smiles: Sequence[str],
    names: Sequence[str],
    batch_size: int,
    config: DescriptorConfig,
) -> Iterator["pd.DataFrame"]:
    """Calculate molecular descriptors in batches to minimize memory usage."""
    total_batches = math.ceil(len(molecules) / batch_size) if molecules else 0
    if total_batches == 0:
        logging.warning("No molecules available for descriptor calculation.")
        return

    batch_iterator = generate_batches(molecules, smiles, names, batch_size)

    for batch_index, (batch_molecules, batch_smiles, batch_names) in enumerate(
        tqdm(batch_iterator, desc="Calculating Descriptors", total=total_batches),
        start=1,
    ):
        if not batch_molecules:
            continue
        
        try:
            # Use single-threaded processing for stability
            batch_dataframe = calculator.pandas(batch_molecules, nproc=1)
            
            # Clean up descriptor data
            skip_cols = [config.smiles_column]
            if config.name_column != config.smiles_column:
                skip_cols.append(config.name_column)
            batch_dataframe = clean_descriptor_data(batch_dataframe, skip_cols)
            
            # Add SMILES and name columns with proper alignment
            batch_dataframe[config.smiles_column] = list(batch_smiles)
            if config.name_column != config.smiles_column:
                batch_dataframe[config.name_column] = list(batch_names)
            
            # Reorder columns to put SMILES/ID first for better readability
            front_cols = [config.smiles_column]
            if config.name_column != config.smiles_column:
                front_cols.append(config.name_column)
            other_cols = [c for c in batch_dataframe.columns if c not in front_cols]
            batch_dataframe = batch_dataframe[front_cols + other_cols]
            
            # Log progress less frequently to reduce noise
            if batch_index % 10 == 1 or batch_index == total_batches:
                logging.info(
                    "Processed batch %d/%d: %d molecules",
                    batch_index,
                    total_batches,
                    len(batch_dataframe),
                )
            else:
                logging.debug(
                    "Processed batch %d/%d: %d molecules",
                    batch_index,
                    total_batches,
                    len(batch_dataframe),
                )
            yield batch_dataframe
        except Exception as e:
            logging.error("Failed to calculate descriptors for batch %d: %s", batch_index, e)
            raise


def write_batches_to_csv(
    batch_dataframes: Iterator["pd.DataFrame"],
    output_path: str,
) -> Tuple[Optional["pd.DataFrame"], int]:
    """Write batch results to CSV file in streaming fashion."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        logging.info("Removing existing output file: %s", output_path)
        output_file.unlink()

    header_written = False
    total_rows = 0
    first_batch: Optional["pd.DataFrame"] = None

    try:
        for batch_dataframe in batch_dataframes:
            if first_batch is None:
                first_batch = batch_dataframe.copy()
            
            batch_mode = "w" if not header_written else "a"
            batch_dataframe.to_csv(
                output_file,
                index=False,
                mode=batch_mode,
                header=not header_written,
            )
            header_written = True
            total_rows += len(batch_dataframe)
    except Exception as e:
        logging.error("Failed to write batch to CSV: %s", e)
        if output_file.exists():
            output_file.unlink()
        raise

    return first_batch, total_rows


def main(config: Optional[DescriptorConfig] = None) -> None:
    """Execute molecular descriptor calculation pipeline."""
    if config is None:
        config = DescriptorConfig()
    
    configure_logging(config.log_level)
    logging.info(
        "Starting descriptor calculation pipeline"
        "\n  Input: %s"
        "\n  Output: %s"
        "\n  Batch size: %d"
        "\n  Encoding: %s"
        "\n  SMILES column: %s"
        "\n  Name column: %s",
        config.input_path,
        config.output_path,
        config.batch_size,
        config.encoding,
        config.smiles_column,
        config.name_column,
    )

    try:
        # Load and validate data
        descriptor_calculator = create_descriptor_calculator(ignore_3d=config.ignore_3d)
        smiles_dataframe = load_smiles_dataframe(config.input_path, config.encoding)
        validate_smiles_dataframe(smiles_dataframe, config.required_columns)

        # Convert SMILES to molecules
        smiles_list = smiles_dataframe[config.smiles_column].tolist()
        name_list = smiles_dataframe[config.name_column].tolist()
        
        # Ensure lists have same length
        if len(smiles_list) != len(name_list):
            raise ValueError(f"SMILES and name lists have different lengths: {len(smiles_list)} vs {len(name_list)}")
        
        molecule_list, filtered_smiles, filtered_names, invalid_entries = convert_smiles_to_molecules(smiles_list, name_list)

        if not molecule_list:
            raise ValueError("No valid molecules found after parsing SMILES input")

        logging.info("Successfully parsed %d/%d molecules", len(molecule_list), len(smiles_list))
        
        # Save invalid SMILES for review if any
        if invalid_entries:
            out_p = Path(config.output_path)
            invalid_path = str(out_p.with_name(out_p.stem + "_invalid_smiles.csv"))
            
            id_col = config.name_column if config.name_column != config.smiles_column else "Name_or_ID"
            invalid_df = pd.DataFrame(invalid_entries, columns=[id_col, config.smiles_column])
            invalid_df.to_csv(invalid_path, index=False)
            logging.info("Saved %d invalid SMILES entries to %s", len(invalid_entries), invalid_path)

        # Calculate descriptors and write results
        batch_generator = calculate_descriptor_batches(
            descriptor_calculator,
            molecule_list,
            filtered_smiles,
            filtered_names,
            config.batch_size,
            config,
        )
        preview_dataframe, total_rows = write_batches_to_csv(batch_generator, config.output_path)

        if preview_dataframe is None:
            raise ValueError("Descriptor calculation did not produce any results")

        # Display preview and summary
        print("\n=== Output Preview ===")
        print(preview_dataframe.head())
        print(f"\nShape: {preview_dataframe.shape}")
        print(f"Columns: {list(preview_dataframe.columns)}")
        
        # Show data quality info
        nan_counts = preview_dataframe.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"\nColumns with NaN values: {nan_counts[nan_counts > 0].to_dict()}")
        
        logging.info(
            "Descriptor calculation completed successfully. %d rows written to %s",
            total_rows,
            config.output_path,
        )
    except Exception as e:
        logging.error("Pipeline failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
