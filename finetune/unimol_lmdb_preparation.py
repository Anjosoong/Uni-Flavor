"""
LMDB Preparation for Uni-Mol Fine-tuning
Generates high-quality molecular conformers for deep learning models
"""

import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set global random seed for reproducibility
np.random.seed(42)

# Configuration parameters
csv_path = r"./odor/train_data_ft.csv"
output_dir = r"./lmdb_preparation_odor"
output_lmdb = "train.lmdb"

# Conformer generation settings
num_3d = 10              # Target number of 3D conformers
num_2d = 1               # Number of 2D conformers as fallback
num_conformers = num_3d + num_2d
num_candidates = 60      # Initial candidates for diversity selection

# Processing parameters
max_atoms = 400          # Maximum atoms per molecule
batch_size = 512        # LMDB batch commit size
map_size = int(2e8)      # LMDB map size (1GB)
nthreads = max(1, mp.cpu_count() - 2)  # CPU threads for multiprocessing


def generate_conformers(mol, target=10):
    """
    Generate high-quality 3D conformers using ETKDGv3 algorithm
    
    Args:
        mol: RDKit molecule object with hydrogens added
        target: Number of conformers to generate
        
    Returns:
        List of 3D coordinates or None if generation fails
    """
    # Configure ETKDGv3 parameters for optimal conformer quality
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = 0.3
    params.useRandomCoords = False      # Ensures reproducible conformers
    params.enforceChirality = True      # Preserves stereochemistry
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    params.numThreads = 1               # Prevents thread conflicts in multiprocessing
    
    # Generate initial conformer ensemble
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_candidates, params=params)
    if len(conf_ids) == 0:
        return None
    
    # Optimize conformers using MMFF force field
    optimization_results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    
    # Extract and validate energies
    energies = []
    for i, conf_id in enumerate(conf_ids):
        if i >= len(optimization_results):
            break
        status, energy = optimization_results[i]
        if energy is not None:  # Filter failed optimizations
            energies.append((conf_id, energy))
    
    if not energies:
        return None
    
    # Sort by energy and apply energy window filtering
    energies.sort(key=lambda x: x[1])
    best_energy = energies[0][1]
    filtered_energies = [(cid, e) for cid, e in energies if e - best_energy <= 5.0]
    
    # Use all conformers if filtering is too restrictive
    if not filtered_energies:
        filtered_energies = energies
    
    # Extract coordinates and center them
    coordinates = []
    for conf_id, _ in filtered_energies:
        coord = mol.GetConformer(conf_id).GetPositions()
        coord = coord - coord.mean(axis=0)  # Center coordinates
        coordinates.append(coord.astype(np.float32))
        
        if len(coordinates) >= target:
            break
    
    if not coordinates:
        return None
    
    # Pad with best conformer if needed
    while len(coordinates) < target:
        coordinates.append(coordinates[0].copy())
    
    return coordinates


def generate_2d(mol):
    """Generate 2D coordinates for molecule as fallback"""
    mol2 = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol2)
    coord = mol2.GetConformer().GetPositions()
    # Center coordinates
    coord = coord - coord.mean(axis=0)
    return coord.astype(np.float32)


def canon_smi(s):
    """Convert SMILES to canonical form"""
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def process_row(row):
    """Process single molecule row to generate high-quality conformers with 2D fallback"""
    smi = row[0]
    target = row[1:]
    
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        
        # Canonicalize SMILES for consistency
        smi = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.AddHs(mol)
        
        # Filter oversized molecules
        if mol.GetNumAtoms() > max_atoms:
            return None
        
        # Generate 3D conformers
        coords_3d = generate_conformers(mol, target=num_3d)
        
        # If 3D generation fails, use 2D fallback
        if coords_3d is None:
            coord_2d = generate_2d(mol)
            if coord_2d is None:
                return None
            # Fill all slots with 2D conformer
            coordinate_list = [coord_2d.copy() for _ in range(num_conformers)]
            num_actual_3d = 0
        else:
            # Add 2D conformer to 3D conformers
            coord_2d = generate_2d(mol)
            if coord_2d is None:
                return None
            coordinate_list = coords_3d + [coord_2d]
            num_actual_3d = len(coords_3d)
        
        # Extract atomic symbols
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        
        # ONLY CHANGE: Use same data format as working code
        payload = {
            'atoms': atoms,
            'coordinates': coordinate_list,  # Keep as list format like working code
            'mol': mol,                      # Include mol object like working code
            'smi': smi,
            'target': target
        }
        
        return pickle.dumps(payload, protocol=4)
    
    except Exception:
        return None


def build_lmdb():
    """Main function to build high-quality LMDB dataset"""
    # Load and preprocess data
    df = pd.read_csv(csv_path)
    
    # Canonical SMILES deduplication
    df["canon"] = df.iloc[:,0].apply(canon_smi)
    df = df.dropna(subset=["canon"])
    df = df.drop_duplicates("canon")
    df.iloc[:,0] = df["canon"]
    df = df.drop(columns=["canon"])
    
    rows = list(df.itertuples(index=False, name=None))
    
    # Setup LMDB environment
    os.makedirs(output_dir, exist_ok=True)
    lmdb_path = os.path.join(output_dir, output_lmdb)
    
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)
    
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1
    )
    
    # Process molecules with multiprocessing
    txn = env.begin(write=True, buffers=True)
    i = 0
    failed = 0
    
    with Pool(nthreads) as pool:
        for result in tqdm(
            pool.imap_unordered(process_row, rows, chunksize=1),
            total=len(rows),
            desc="Generating high-quality conformers"
        ):
            if result is None:
                failed += 1
                continue
            
            txn.put(str(i).encode(), result)
            i += 1
            
            # Commit in batches for better performance
            if i > 0 and i % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True, buffers=True)
    
    txn.commit()
    env.close()
    
    print(f"Successfully processed {i} molecules, failed: {failed}")


if __name__ == "__main__":
    mp.freeze_support()
    build_lmdb()