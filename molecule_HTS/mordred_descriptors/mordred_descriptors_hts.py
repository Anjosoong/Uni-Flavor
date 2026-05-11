#!/usr/bin/env python3
"""
Molecular Descriptor Calculation Pipeline using Mordred
Optimized for HTS multi-core CPU environments - COCONUT Database Processing

Batch processing pipeline for calculating molecular descriptors from SMILES strings.
Memory-efficient implementation suitable for large molecular datasets (730,000+ molecules).

Dependencies: pandas, mordred, rdkit, tqdm
"""

import os
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm


@dataclass
class DescriptorConfig:
    """Configuration for molecular descriptor calculation pipeline."""
    
    input_path: str = "coconut_calculation.csv"
    output_dir: str = "mordred_output"
    output_path: str = "coconut_mordred_descriptors.csv"
    batch_size: int = 512  # Batch size for 32-core CPU
    nproc: int = 32
    encoding: str = "utf-8"
    log_level: int = logging.INFO
    ignore_3d: bool = True
    smiles_column: str = "SMILES"
    id_column: str = "id"
    save_checkpoint: bool = True
    checkpoint_interval: int = 10  # Save every 10 batches
    save_numpy: bool = True  # Save as .npy in addition to CSV
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.nproc <= 0:
            raise ValueError(f"nproc must be positive, got {self.nproc}")
        if not self.input_path:
            raise ValueError("Input path cannot be empty")
        if not self.output_path:
            raise ValueError("Output path cannot be empty")
        if not self.output_dir:
            raise ValueError("Output directory cannot be empty")


def resolve_output_paths(config: DescriptorConfig) -> DescriptorConfig:
    """Resolve output file into the configured output directory."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = Path(config.output_path).name
    config.output_dir = str(output_dir)
    config.output_path = str(output_dir / output_name)
    return config


def configure_logging(log_level: int = logging.INFO, log_file: str = None) -> None:
    """Configure logging with consistent format."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )


def create_descriptor_calculator(ignore_3d: bool = True) -> Calculator:
    """Create Mordred calculator for 2D molecular descriptors."""
    return Calculator(descriptors, ignore_3D=ignore_3d)


def load_smiles_dataframe(csv_path: str, encoding: str) -> pd.DataFrame:
    """Load SMILES dataset from CSV file with encoding fallback."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    
    encodings_to_try = [encoding, "utf-8-sig", "utf-8", "gbk", "cp1252"]
    encodings_to_try = list(dict.fromkeys(encodings_to_try))  # Remove duplicates
    
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
            if enc == encodings_to_try[-1]:
                raise ValueError(f"Could not decode file with any of these encodings: {encodings_to_try}")
            continue
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
    
    raise ValueError(f"Could not load file with any encoding: {encodings_to_try}")


def validate_smiles_dataframe(dataframe: pd.DataFrame, smiles_col: str, id_col: str) -> pd.DataFrame:
    """Validate DataFrame contains required columns and auto-generate ID if missing."""
    # Check SMILES column (required)
    if smiles_col not in dataframe.columns:
        raise ValueError(f"DataFrame must contain SMILES column: {smiles_col}")
    
    # Auto-generate ID column if missing
    if id_col not in dataframe.columns:
        logging.warning(f"ID column '{id_col}' not found. Auto-generating sequential IDs.")
        dataframe = dataframe.copy()
        dataframe.insert(0, id_col, [f"mol_{i:07d}" for i in range(len(dataframe))])
        logging.info(f"Generated {len(dataframe)} IDs: mol_0000000 to mol_{len(dataframe)-1:07d}")
    
    return dataframe


def convert_smiles_to_molecules(
    smiles: Sequence[str],
    ids: Sequence[str],
) -> Tuple[List[Mol], List[str], List[str], List[Tuple[str, str]]]:
    """Convert SMILES strings to RDKit molecule objects, filtering invalid entries."""
    valid_molecules: List[Mol] = []
    valid_smiles: List[str] = []
    valid_ids: List[str] = []
    invalid_entries: List[Tuple[str, str]] = []

    for smiles_string, mol_id in zip(smiles, ids):
        if pd.isna(smiles_string):
            invalid_entries.append((str(mol_id), str(smiles_string)))
            continue
        
        smiles_string = str(smiles_string).strip()
        if not smiles_string:
            invalid_entries.append((str(mol_id), smiles_string))
            continue
        
        try:
            molecule = Chem.MolFromSmiles(smiles_string)
            if molecule is None:
                invalid_entries.append((str(mol_id), smiles_string))
                continue
            valid_molecules.append(molecule)
            valid_smiles.append(smiles_string)
            valid_ids.append(str(mol_id))
        except Exception as e:
            logging.debug("Failed to parse SMILES '%s' for ID %s: %s", smiles_string, mol_id, e)
            invalid_entries.append((str(mol_id), smiles_string))

    if invalid_entries:
        sample_entries = ", ".join(
            f"{entry_id}:{entry_smiles[:20]}" for entry_id, entry_smiles in invalid_entries[:5]
        )
        logging.warning(
            "Skipped %d invalid SMILES entries (%.2f%%). Examples: %s",
            len(invalid_entries),
            100 * len(invalid_entries) / len(smiles) if smiles else 0,
            sample_entries,
        )

    return valid_molecules, valid_smiles, valid_ids, invalid_entries


def generate_batches(
    molecules: Sequence[Mol],
    smiles: Sequence[str],
    ids: Sequence[str],
    batch_size: int,
) -> Iterator[Tuple[Sequence[Mol], Sequence[str], Sequence[str]]]:
    """Generate fixed-size batches of molecules for memory-efficient processing."""
    for start_index in range(0, len(molecules), batch_size):
        end_index = start_index + batch_size
        yield (
            molecules[start_index:end_index], 
            smiles[start_index:end_index], 
            ids[start_index:end_index]
        )


def clean_descriptor_data(dataframe: pd.DataFrame, skip_columns: Sequence[str]) -> pd.DataFrame:
    """Clean descriptor data by handling inf values and converting descriptor columns to numeric."""
    dataframe = dataframe.replace([float("inf"), float("-inf")], float("nan"))
    
    descriptor_cols = [c for c in dataframe.columns if c not in set(skip_columns)]
    obj_cols = [c for c in descriptor_cols if dataframe[c].dtype == "object"]
    if obj_cols:
        dataframe[obj_cols] = dataframe[obj_cols].apply(pd.to_numeric, errors="coerce")
    
    return dataframe


def calculate_descriptor_batches(
    calculator: Calculator,
    molecules: Sequence[Mol],
    smiles: Sequence[str],
    ids: Sequence[str],
    config: DescriptorConfig,
) -> Iterator[pd.DataFrame]:
    """Calculate molecular descriptors in batches to minimize memory usage."""
    total_batches = math.ceil(len(molecules) / config.batch_size) if molecules else 0
    if total_batches == 0:
        logging.warning("No molecules available for descriptor calculation.")
        return

    batch_iterator = generate_batches(molecules, smiles, ids, config.batch_size)

    for batch_index, (batch_molecules, batch_smiles, batch_ids) in enumerate(
        tqdm(batch_iterator, desc="Calculating Descriptors", total=total_batches),
        start=1,
    ):
        if not batch_molecules:
            continue
        
        try:
            # Use multi-threaded processing for maximum performance
            batch_dataframe = calculator.pandas(batch_molecules, nproc=config.nproc)
            
            # Clean up descriptor data
            skip_cols = [config.smiles_column, config.id_column]
            batch_dataframe = clean_descriptor_data(batch_dataframe, skip_cols)
            
            # Add SMILES and ID columns
            batch_dataframe[config.id_column] = list(batch_ids)
            batch_dataframe[config.smiles_column] = list(batch_smiles)
            
            # Reorder columns to put ID and SMILES first
            front_cols = [config.id_column, config.smiles_column]
            other_cols = [c for c in batch_dataframe.columns if c not in front_cols]
            batch_dataframe = batch_dataframe[front_cols + other_cols]
            
            # Log progress
            if batch_index % 10 == 1 or batch_index == total_batches:
                logging.info(
                    "Processed batch %d/%d: %d molecules (%.1f%% complete)",
                    batch_index,
                    total_batches,
                    len(batch_dataframe),
                    100 * batch_index / total_batches
                )
            
            yield batch_dataframe
        except Exception as e:
            logging.error("Failed to calculate descriptors for batch %d: %s", batch_index, e)
            raise


def write_batches_to_csv(
    batch_dataframes: Iterator[pd.DataFrame],
    output_path: str,
    checkpoint_interval: int = 10,
    save_numpy: bool = True,
    n_molecules: int = 0,
    id_column: str = "id",
    smiles_column: str = "smiles",
) -> Tuple[Optional[pd.DataFrame], int, Optional[List[str]], List[str], List[str]]:
    """Write batch results to CSV/NPY file in streaming fashion with checkpoints.

    Uses numpy memmap for incremental writing to avoid accumulating all data in RAM.
    Returns (first_batch, total_rows, descriptor_cols, meta_ids, meta_smiles).
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        logging.info("Removing existing output file: %s", output_path)
        output_file.unlink()

    header_written = False
    total_rows = 0
    first_batch: Optional[pd.DataFrame] = None
    batch_count = 0

    # Numpy incremental write via memmap (avoids accumulating all batches in RAM)
    npy_memmap = None
    npy_row_offset = 0
    descriptor_cols: Optional[List[str]] = None
    meta_ids: List[str] = []
    meta_smiles: List[str] = []
    output_base = Path(output_path).stem
    output_dir = Path(output_path).parent
    npy_path = output_dir / f"{output_base}_features.npy"

    try:
        for batch_dataframe in batch_dataframes:
            if first_batch is None:
                first_batch = batch_dataframe.copy()

            # Save to CSV
            batch_mode = "w" if not header_written else "a"
            batch_dataframe.to_csv(
                output_file,
                index=False,
                mode=batch_mode,
                header=not header_written,
            )
            header_written = True
            total_rows += len(batch_dataframe)
            batch_count += 1

            # Incremental numpy write via memmap
            if save_numpy:
                if descriptor_cols is None:
                    descriptor_cols = [c for c in batch_dataframe.columns
                                       if c not in (id_column, smiles_column)]
                    n_rows = n_molecules if n_molecules > 0 else 1_000_000
                    npy_memmap = np.lib.format.open_memmap(
                        str(npy_path), mode='w+', dtype=np.float64,
                        shape=(n_rows, len(descriptor_cols))
                    )
                batch_data = batch_dataframe[descriptor_cols].values.astype(np.float64)
                end = npy_row_offset + len(batch_data)
                npy_memmap[npy_row_offset:end] = batch_data
                npy_row_offset = end
                meta_ids.extend(batch_dataframe[id_column].tolist())
                meta_smiles.extend(batch_dataframe[smiles_column].tolist())

            # Save checkpoint
            if checkpoint_interval > 0 and batch_count % checkpoint_interval == 0:
                logging.info("Checkpoint: %d rows written", total_rows)

    except Exception as e:
        logging.error("Failed to write batch to CSV: %s", e)
        raise

    # Finalize memmap: trim to actual row count if pre-allocation was larger
    if save_numpy and npy_memmap is not None:
        if npy_row_offset < npy_memmap.shape[0]:
            logging.info("Trimming numpy array from %d to %d rows", npy_memmap.shape[0], npy_row_offset)
            final_array = np.array(npy_memmap[:npy_row_offset])
            del npy_memmap
            np.save(str(npy_path), final_array)
            del final_array
        else:
            npy_memmap.flush()
            del npy_memmap
        logging.info("Numpy feature array written: %s (%d x %d)", npy_path, npy_row_offset, len(descriptor_cols))

    return first_batch, total_rows, descriptor_cols, meta_ids, meta_smiles


def main(config: Optional[DescriptorConfig] = None) -> None:
    """Execute molecular descriptor calculation pipeline."""
    if config is None:
        config = DescriptorConfig()
    config = resolve_output_paths(config)
     
    # Setup logging
    log_file = str(Path(config.output_path).with_suffix('.log'))
    configure_logging(config.log_level, log_file)
    
    logging.info("Mordred Descriptor Calculation Pipeline - HTS Optimized")
    logging.info(
        "Configuration: input=%s | output_dir=%s | output=%s | "
        "batch_size=%d | nproc=%d | encoding=%s | smiles_col=%s | id_col=%s",
        config.input_path,
        config.output_dir,
        config.output_path,
        config.batch_size,
        config.nproc,
        config.encoding,
        config.smiles_column,
        config.id_column,
    )

    start_time = datetime.now()

    try:
        # Load and validate data
        logging.info("[1/4] Loading data...")
        descriptor_calculator = create_descriptor_calculator(ignore_3d=config.ignore_3d)
        smiles_dataframe = load_smiles_dataframe(config.input_path, config.encoding)
        smiles_dataframe = validate_smiles_dataframe(smiles_dataframe, config.smiles_column, config.id_column)

        # Convert SMILES to molecules
        logging.info("[2/4] Parsing SMILES...")
        smiles_list = smiles_dataframe[config.smiles_column].tolist()
        id_list = smiles_dataframe[config.id_column].tolist()
        
        molecule_list, filtered_smiles, filtered_ids, invalid_entries = convert_smiles_to_molecules(
            smiles_list, id_list
        )

        if not molecule_list:
            raise ValueError("No valid molecules found after parsing SMILES input")

        logging.info("Successfully parsed %d/%d molecules (%.2f%% success rate)", 
                    len(molecule_list), len(smiles_list),
                    100 * len(molecule_list) / len(smiles_list))
        
        # Save invalid SMILES for review if any
        if invalid_entries:
            out_p = Path(config.output_path)
            invalid_path = str(out_p.with_name(out_p.stem + "_invalid_smiles.csv"))
            invalid_df = pd.DataFrame(invalid_entries, columns=[config.id_column, config.smiles_column])
            invalid_df.to_csv(invalid_path, index=False)
            logging.info("Saved %d invalid SMILES entries to %s", len(invalid_entries), invalid_path)

        # Calculate descriptors and write results
        logging.info("[3/4] Calculating Mordred descriptors...")
        logging.info("Estimated time: %.1f - %.1f hours (based on 0.5-1 sec per molecule)",
                    len(molecule_list) / config.nproc / 3600 * 0.5,
                    len(molecule_list) / config.nproc / 3600 * 1.0)
        
        batch_generator = calculate_descriptor_batches(
            descriptor_calculator,
            molecule_list,
            filtered_smiles,
            filtered_ids,
            config,
        )
        
        logging.info("[4/4] Writing results...")
        preview_dataframe, total_rows, descriptor_cols, meta_ids, meta_smiles = write_batches_to_csv(
            batch_generator,
            config.output_path,
            config.checkpoint_interval,
            config.save_numpy,
            n_molecules=len(molecule_list),
            id_column=config.id_column,
            smiles_column=config.smiles_column,
        )

        if preview_dataframe is None:
            raise ValueError("Descriptor calculation did not produce any results")

        # Save numpy metadata files (npy array already written incrementally by write_batches_to_csv)
        if config.save_numpy and descriptor_cols is not None:
            logging.info("[5/5] Saving numpy metadata...")
            output_base = Path(config.output_path).stem
            output_dir = Path(config.output_path).parent
            npy_path = output_dir / f"{output_base}_features.npy"

            # Save metadata (ID, SMILES)
            metadata_df = pd.DataFrame({config.id_column: meta_ids, config.smiles_column: meta_smiles})
            metadata_path = output_dir / f"{output_base}_metadata.csv"
            metadata_df.to_csv(metadata_path, index=False)
            logging.info("Metadata saved: %s", metadata_path)

            # Save descriptor names
            descriptor_names_path = output_dir / f"{output_base}_descriptor_names.txt"
            with open(descriptor_names_path, 'w') as f:
                f.write('\n'.join(descriptor_cols))
            logging.info("Descriptor names saved: %s (%d descriptors)", descriptor_names_path, len(descriptor_cols))

            # Save alignment info
            alignment_info = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_molecules': total_rows,
                'total_descriptors': len(descriptor_cols),
                'id_column': config.id_column,
                'smiles_column': config.smiles_column,
                'files': {
                    'features': str(npy_path.name),
                    'metadata': str(metadata_path.name),
                    'descriptor_names': str(descriptor_names_path.name),
                    'full_csv': str(Path(config.output_path).name)
                },
                'alignment_note': 'Row i in features.npy corresponds to row i in metadata.csv'
            }
            alignment_path = output_dir / f"{output_base}_alignment_info.json"
            with open(alignment_path, 'w') as f:
                json.dump(alignment_info, f, indent=2)
            logging.info("Alignment info saved: %s", alignment_path)

        # Calculate elapsed time
        elapsed_time = datetime.now() - start_time
        
        # Display summary
        logging.info("Calculation completed successfully.")
        print("")
        print(f"Total molecules processed: {total_rows:,}")
        print(f"Total descriptors: {len(preview_dataframe.columns) - 2}")  # Exclude ID and SMILES
        print(f"Elapsed time: {elapsed_time}")
        print(f"Average speed: {total_rows / elapsed_time.total_seconds():.2f} molecules/second")
        print(f"\nOutput files:")
        print(f"  CSV (full):     {config.output_path}")
        if config.save_numpy:
            output_base = Path(config.output_path).stem
            output_dir = Path(config.output_path).parent
            print(f"  NPY (features): {output_dir / (output_base + '_features.npy')}")
            print(f"  Metadata:       {output_dir / (output_base + '_metadata.csv')}")
            print(f"  Descriptor names: {output_dir / (output_base + '_descriptor_names.txt')}")
            print(f"  Alignment info: {output_dir / (output_base + '_alignment_info.json')}")
        print(f"  Log file:       {log_file}")
        
        # Show data quality info
        nan_counts = preview_dataframe.isnull().sum()
        if nan_counts.sum() > 0:
            high_nan = nan_counts[nan_counts > len(preview_dataframe) * 0.5]
            if not high_nan.empty:
                print(f"\nWarning: {len(high_nan)} descriptors have >50% NaN values")
        
        logging.info(
            "Pipeline completed successfully. %d rows written to %s in %s",
            total_rows,
            config.output_path,
            elapsed_time
        )
        
    except Exception as e:
        logging.error("Pipeline failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Mordred descriptors for COCONUT database (HTS optimized)"
    )
    parser.add_argument("--input", type=str, default="coconut_calculation.csv",
                       help="Input CSV file path")
    parser.add_argument("--output-dir", type=str, default="mordred_output",
                       help="Directory to store CSV, log, invalid SMILES, and NumPy outputs")
    parser.add_argument("--output", type=str, default="coconut_mordred_descriptors.csv",
                       help="Output CSV file path")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size for processing")
    parser.add_argument("--nproc", type=int, default=32,
                       help="Number of CPU cores to use (default: 32)")
    parser.add_argument("--smiles-col", type=str, default="SMILES",
                       help="SMILES column name (default: canonical_smiles)")
    parser.add_argument("--id-col", type=str, default="id",
                       help="ID column name (default: id)")
    parser.add_argument("--no-numpy", action="store_true",
                       help="Disable numpy output (only save CSV)")
    
    args = parser.parse_args()
    
    config = DescriptorConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        nproc=args.nproc,
        smiles_column=args.smiles_col,
        id_column=args.id_col,
        save_numpy=not args.no_numpy,
    )
    
    main(config)
