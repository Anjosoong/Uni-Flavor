#!/usr/bin/env python3
"""
SMILES to MOL2 converter for protein-ligand docking.
Supports AutoDock Vina, AutoDock 4, GOLD, Glide, etc.

Usage:
    python smiles_to_mol2_docking.py --smiles "CC(=CCC/C(=C/CO)/C)C" --name Geraniol
    python smiles_to_mol2_docking.py --input ligands.csv --smiles_col SMILES --name_col Name
    python smiles_to_mol2_docking.py --input ligands.txt
"""

import os
import hashlib
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("ERROR: RDKit not installed. Run: pip install rdkit")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configuration
CONFIG = {
    'output_dir':          r'./mol2_output',
    'force_field':         'MMFF94',   # 'MMFF94' (recommended) or 'UFF'
    'max_iterations':      500,         # energy minimization steps
    'add_hydrogens':       True,        # required for docking
    'assign_charges':      True,        # Gasteiger charges, required for docking
    'generate_conformers': False,       # generate multiple conformers
    'num_conformers':      10,          # number of conformers if enabled
    'random_seed':         42,
}


class DockingLigandPreparation:
    """Ligand preparation for protein-ligand docking."""

    def __init__(self, output_dir: str = CONFIG['output_dir']):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.force_field         = CONFIG['force_field']
        self.max_iterations      = CONFIG['max_iterations']
        self.add_hydrogens       = CONFIG['add_hydrogens']
        self.assign_charges      = CONFIG['assign_charges']
        self.generate_conformers = CONFIG['generate_conformers']
        self.num_conformers      = CONFIG['num_conformers']
        self.random_seed         = CONFIG['random_seed']

    def _mol_to_mol2_format(self, mol, mol_name: str) -> str:
        """Manually build MOL2 string (fallback if Chem.MolToMol2File unavailable)."""
        conf = mol.GetConformer()
        lines = [
            "@<TRIPOS>MOLECULE", mol_name,
            f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0",
            "SMALL", "GASTEIGER", "",
            "@<TRIPOS>ATOM",
        ]
        for i, atom in enumerate(mol.GetAtoms(), 1):
            pos = conf.GetAtomPosition(atom.GetIdx())
            charge = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
            sym = atom.GetSymbol()
            lines.append(f"{i:7d} {sym:4s} {pos.x:10.4f} {pos.y:10.4f} {pos.z:10.4f} {sym:5s} 1 {mol_name:8s} {charge:10.4f}")
        lines.append("@<TRIPOS>BOND")
        _bmap = {Chem.BondType.DOUBLE: "2", Chem.BondType.TRIPLE: "3", Chem.BondType.AROMATIC: "ar"}
        for i, bond in enumerate(mol.GetBonds(), 1):
            bt = _bmap.get(bond.GetBondType(), "1")
            lines.append(f"{i:6d} {bond.GetBeginAtomIdx()+1:5d} {bond.GetEndAtomIdx()+1:5d} {bt:4s}")
        return "\n".join(lines)

    def _assign_charges(self, mol):
        try:
            AllChem.ComputeGasteigerCharges(mol)
            logging.info("  + Gasteiger charges assigned")
        except Exception as e:
            logging.warning(f"  ! Gasteiger charge failed: {e}")
        return mol

    def _generate_conformers(self, mol):
        """Generate and minimize multiple conformers; return (mol, sorted energies)."""
        conf_ids = AllChem.EmbedMultipleConfs(
            mol, numConfs=self.num_conformers,
            randomSeed=self.random_seed, pruneRmsThresh=0.5,
        )
        if not conf_ids:
            return mol, []
        energies = []
        for cid in conf_ids:
            if self.force_field == 'MMFF94':
                props = AllChem.MMFFGetMoleculeProperties(mol)
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid) if props else None
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if ff:
                ff.Initialize(); ff.Minimize(maxIts=self.max_iterations)
                energies.append((cid, ff.CalcEnergy()))
        energies.sort(key=lambda x: x[1])
        logging.info(f"  + {len(energies)} conformers generated, lowest E = {energies[0][1]:.2f} kcal/mol")
        return mol, energies

    def smiles_to_mol2(self, smiles: str, mol_name: str = None,
                       optimize: bool = True) -> Tuple[bool, str]:
        """Convert one SMILES to MOL2. Returns (success, output_path_or_error)."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, f"Invalid SMILES: {smiles}"

            if self.add_hydrogens:
                mol = Chem.AddHs(mol)
                logging.info("  + Hydrogens added")

            # 3D embedding
            if self.generate_conformers and self.num_conformers > 1:
                mol, energies = self._generate_conformers(mol)
                if not energies:
                    if AllChem.EmbedMolecule(mol, randomSeed=self.random_seed) != 0:
                        if AllChem.EmbedMolecule(mol, randomSeed=self.random_seed, useRandomCoords=True) != 0:
                            return False, f"3D embedding failed: {smiles}"
                    logging.info("  + 3D coords generated (single conformer fallback)")
            else:
                if AllChem.EmbedMolecule(mol, randomSeed=self.random_seed) != 0:
                    if AllChem.EmbedMolecule(mol, randomSeed=self.random_seed, useRandomCoords=True) != 0:
                        return False, f"3D embedding failed: {smiles}"
                logging.info("  + 3D coords generated")

            # Energy minimization
            if optimize:
                if self.force_field == 'MMFF94':
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    if props is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                        ff.Initialize()
                        converged = ff.Minimize(maxIts=self.max_iterations)
                        energy = ff.CalcEnergy()
                        status = "converged" if converged == 0 else "not fully converged"
                        logging.info(f"  + MMFF94 minimization {status} (E = {energy:.2f} kcal/mol)")
                    else:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=self.max_iterations)
                        logging.info("  + UFF minimization done (MMFF94 unavailable)")
                else:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=self.max_iterations)
                    logging.info("  + UFF minimization done")

            if self.assign_charges:
                mol = self._assign_charges(mol)

            if mol_name is None:
                mol_name = hashlib.md5(smiles.encode()).hexdigest()[:10]
            mol_name = "".join(c for c in mol_name if c.isalnum() or c in ('-', '_'))

            output_file = self.output_dir / f"{mol_name}.mol2"
            try:
                Chem.MolToMol2File(mol, str(output_file))
            except (AttributeError, TypeError):
                with open(output_file, 'w') as f:
                    f.write(self._mol_to_mol2_format(mol, mol_name))

            if output_file.exists():
                logging.info(f"  + Saved: {output_file} ({output_file.stat().st_size} bytes)")
                return True, str(output_file)
            return False, f"File not written: {mol_name}"

        except Exception as e:
            return False, f"Conversion error: {e}"

    def convert_single(self, smiles: str, mol_name: str = None) -> bool:
        logging.info(f"SMILES : {smiles}")
        if mol_name:
            logging.info(f"Name   : {mol_name}")
        success, result = self.smiles_to_mol2(smiles, mol_name)
        if success:
            logging.info(f"Output : {result}")
        else:
            logging.error(f"Failed : {result}")
        return success

    def convert_batch(self, input_file: str,
                      smiles_col: Optional[str] = None,
                      name_col: Optional[str] = None):
        input_path = Path(input_file)
        if not input_path.exists():
            logging.error(f"File not found: {input_file}")
            return

        smiles_list: List[Tuple[str, str]] = []

        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
            # auto-detect SMILES column
            col = smiles_col or next((c for c in df.columns if 'smiles' in c.lower()), None)
            if col is None:
                logging.error("No SMILES column found. Use --smiles_col to specify.")
                return
            for idx, row in df.iterrows():
                name = str(row[name_col]) if name_col and name_col in df.columns else f"ligand_{idx+1}"
                smiles_list.append((row[col], name))
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ',' in line:
                        parts = [p.strip() for p in line.split(',', 1)]
                        # heuristic: if first part looks like SMILES
                        if any(c in parts[0] for c in ['C', 'N', 'O', '(', '=']):
                            smiles_list.append((parts[0], parts[1] if len(parts) > 1 else f"ligand_{idx}"))
                        else:
                            smiles_list.append((parts[1], parts[0]) if len(parts) > 1 else (parts[0], f"ligand_{idx}"))
                    else:
                        smiles_list.append((line, f"ligand_{idx}"))

        logging.info(f"Loaded {len(smiles_list)} SMILES from {input_file}")
        logging.info(f"Output dir: {self.output_dir}")

        ok, fail, failures = 0, 0, []
        for idx, (smi, name) in enumerate(smiles_list, 1):
            logging.info(f"\n[{idx}/{len(smiles_list)}] {name}  |  {smi}")
            success, result = self.smiles_to_mol2(smi, name)
            if success:
                ok += 1
            else:
                fail += 1
                failures.append((name, result))
                logging.error(f"  ! {result}")

        logging.info(f"Done — success: {ok}  failed: {fail}  total: {len(smiles_list)}")
        logging.info(f"Output dir: {self.output_dir}")
        if failures:
            logging.info("Failed molecules:")
            for name, reason in failures:
                logging.info(f"  - {name}: {reason}")


def main():
    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit required. Install: conda install -c conda-forge rdkit")
        return

    parser = argparse.ArgumentParser(
        description="SMILES to MOL2 converter for protein-ligand docking"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--smiles',  type=str, help='Single SMILES string')
    grp.add_argument('--input',   type=str, help='Input file (.csv or .txt)')
    parser.add_argument('--name',       type=str, default=None, help='Molecule name (single mode)')
    parser.add_argument('--smiles_col', type=str, default=None, help='SMILES column name in CSV')
    parser.add_argument('--name_col',   type=str, default=None, help='Name column in CSV')
    parser.add_argument('--output_dir', type=str, default=CONFIG['output_dir'], help='Output directory')
    args = parser.parse_args()

    prep = DockingLigandPreparation(output_dir=args.output_dir)

    if args.smiles:
        prep.convert_single(args.smiles, args.name)
    else:
        prep.convert_batch(args.input, smiles_col=args.smiles_col, name_col=args.name_col)


if __name__ == '__main__':
    main()
