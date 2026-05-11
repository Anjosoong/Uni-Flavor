"""
Functional group definitions for molecular analysis.

Three dictionaries are exported and imported by analyze_molecular_groups.py
and visualize_molecular_groups.py:

  group_priorities    - maps priority level ('high'/'medium'/'low') to an ordered
                        list of group names.  Higher-priority groups are matched
                        first so that composite groups (e.g. Ester) are not
                        double-counted as their constituent parts (e.g. Ketone).

  group_design_smiles - maps group name to a human-readable design template showing
                        attachment points (*).  Included in output CSV files as a
                        molecular design reference.

  group_patterns      - maps group name to the RDKit SMARTS pattern used for
                        substructure matching during detection.
"""


# Detection priority: composite groups first, then elementary, then local units
group_priorities = {
    # Priority 1: Composite functional groups
    'high': [
        'Carboxyl',
        'Amide',
        'Ester',
        'Carbonate',
        'Carbamate Ester',
        'Acyl Chloride',
        'Sulfonic',
        'Phosphoric',
        'Sulfonamide',
        'Lactone',
        'Lactam',
        'Anhydride',
        'Isocyanate',
        'Polypeptide',
    ],
    
    # Priority 2: Elementary functional groups
    'medium': [
        'Aldehyde',
        'Ketone',
        'Primary Amine',
        'Secondary Amine',
        'Tertiary Amine',
        'Hydroxyl',
        'Thiol',
        'Ether',
        'Nitro',
        'Cyano',
        'Nitroso',
        'Oxime',
        'Vinyl Ether',
        'Acrylate',
    ],
    
    # Priority 3: Local structural units
    'low': [
        'Double Bond (C=C)',
        'Triple Bond (C#C)',
        'Benzene',
        'Pyridine',
        'Pyrrole',
        'Furan',
        'Thiophene',
        'Imidazole',
        'Thiazole',
        'Xylene',
        'Diphenyl',
        'Fluorine',
        'Chlorine',
        'Bromine',
        'Iodine',
        'Trifluoromethyl',
        'Halogenated Aromatic',
        'Vinyl Halide',
        'Alkyl',
        'Alkene',
        'Alkyne',
        'Allyl',
        'Vinyl',
        'Phenyl',
    ]
}

group_design_smiles = {
    # Composite functional groups
    'Carboxyl':         '*C(=O)O',
    'Ester':            '*C(=O)O*',
    'Amide':            '*C(=O)N*',
    'Anhydride':        '*C(=O)OC(=O)*',
    'Sulfonic':         '*S(=O)(=O)O',
    'Phosphoric':       '*P(=O)(O)O',
    'Lactone':          '*C(=O)O* (ring)',
    'Lactam':           '*C(=O)N* (ring)',
    'Carbonate':        '*OC(=O)O*',
    'Sulfonamide':      '*S(=O)(=O)N*',
    'Sulfonate Ester':  '*S(=O)(=O)O*',
    'Phosphate Ester':  '*OP(=O)(O)O*',
    'Phosphoryl Ester': '*OP(=O)(O)O*',
    'Carbamate Ester':  '*NC(=O)O*',
    'Acyl Chloride':    '*C(=O)Cl',
    'Isocyanate':       '*N=C=O',
    'Polypeptide':      '*NH-*CH-C(=O)-NH-*CH-C(=O)*',

    # Elementary functional groups
    'Primary Amine':    '*NH2',
    'Secondary Amine':  '*NH*',
    'Tertiary Amine':   '*N(*)*',
    'Hydroxyl':         '*O',
    'Aldehyde':         '*C=O',
    'Ketone':           '*C(=O)*',
    'Thiol':            '*S',
    'Ether':            '*O*',
    'Nitro':            '*[N+](=O)[O-]',
    'Cyano':            '*C#N',
    'Nitroso':          '*N=O',
    'Hydrazine':        '*NHN*',
    'Oxime':            '*C=NO',
    'Diazo':            '*N=N*',
    'Aromatic Amine':   '*c-N*',
    'Diamine':          '*NH-C-NH*',
    'Amino Acid':       '*NH2-*CH-C(=O)O',
    'Glycol':           '*C(O)O',
    'Epoxide':          '*C1OC1*',
    'Vinyl Ether':      '*C=CO*',
    'Ether Alcohol':    '*O-C-O*',
    'Acrylate':         '*C=CC(=O)O*',

    # Aromatic ring systems
    'Benzene':          '*c1ccccc1',
    'Pyridine':         '*c1ccccn1',
    'Pyrrole':          '*c1cc[nH]c1',
    'Furan':            '*c1ccco1',
    'Thiophene':        '*c1cccs1',
    'Imidazole':        '*c1c[nH]cn1',
    'Thiazole':         '*c1cscn1',
    'Xylene':           '*c1cc(C)cc(C)c1',
    'Diphenyl':         '*c1ccccc1-c2ccccc2',

    # Halogen substituents
    'Fluorine':             '*F',
    'Chlorine':             '*Cl',
    'Bromine':              '*Br',
    'Iodine':               '*I',
    'Trifluoromethyl':      '*C(F)(F)F',
    'Halogenated Aromatic': '*c1ccccc1-[X] (X=F/Cl/Br/I)',
    'Vinyl Halide':         '*C=C-[X] (X=F/Cl/Br/I)',

    # Sulfur-containing
    'Sulfonyl':  '*S(=O)(=O)*',
    'Thionone':  '*C=S',

    # Alkyl / alkenyl groups
    'Alkyl':             '*C',
    'Alkene':            '*C=C*',
    'Alkyne':            '*C#C*',
    'Double Bond (C=C)': '*C=C*',
    'Triple Bond (C#C)': '*C#C*',
    'Butene':            '*C=CCC*',
    'Pentene':           '*C=CCCC*',
    'Allyl':             '*CC=C',
    'Vinyl':             '*C=C',
    'Phenyl':            '*c1ccccc1',

    # Miscellaneous
    'Silane':              '*[Si](*)(*)(*)',
    'Alkylamide':          '*C-N-C(=O)*',
    'Substituted Alkene':  '*C(*)=C(*)*',
    'Acetate':             '*OC(=O)C',
    'Nitrate Ester':       '*O[N+](=O)[O-]',
    'Halohydrin':          '*C(O)-C-[X] (X=F/Cl/Br/I)',
}

group_patterns = {
    # Composite functional groups
    'Carboxyl': '[C](=[O])[OH]',
    'Ester': '[C](=[O])[O;!R]-[C]',
    'Amide': '[C](=[O])[N]',
    'Anhydride': '[C](=[O])[O][C](=[O])',
    'Sulfonic': '[S](=O)(=O)[OH]',
    'Phosphoric': '[P](=O)([OH])[OH]',
    'Lactone': '[C](=[O])[O;R]',
    'Lactam': '[C](=[O])[N;R]',
    'Carbonate': '[O]-[C](=O)[O]',
    'Sulfonamide': '[S](=O)(=O)[N]',
    'Sulfonate Ester': '[S](=O)(=O)[O]-[C]',
    'Phosphate Ester': '[P](=O)([O]-[C])([O])',
    'Phosphoryl Ester': '[P](=O)([O]-[C])([O])',
    'Carbamate Ester': '[N]-[C](=O)[O]-[C]',
    'Acyl Chloride': '[C](=O)[Cl]',
    'Isocyanate': '[N]=[C]=[O]',
    'Polypeptide': '[NH2]-[CH]-[C](=[O])-[NH]-[CH]-[C](=[O])',
    
    # Elementary functional groups
    'Primary Amine': '[NX3H2][CX4]',  # R-NH2: one carbon substituent, two hydrogen atoms
    'Secondary Amine': '[NX3H1]([CX4])[CX4]',  # R1-NH-R2: two carbon substituents, one hydrogen atom
    'Tertiary Amine': '[NX3H0]([CX4])([CX4])[CX4]',  # R1-N(R2)-R3: three carbon substituents, no hydrogen atoms
    'Hydroxyl': '[OH]',
    'Aldehyde': '[CX3H1](=O)[#6]',  # Precise aldehyde pattern requiring one hydrogen and one carbon neighbor
    'Ketone': '[CX3](=[O])[CX4]',
    'Thiol': '[SH]',
    'Ether': '[C]-[O;!$(O=C)]-[C]',
    'Nitro': '[N+0](=[O])[O-]',
    'Cyano': '[C]#[N]',
    'Nitroso': '[N]=[O]',
    'Hydrazine': '[NH2]-[NH2]',
    'Oxime': '[C]=[N]-[OH]',
    'Diazo': '[N]=[N]',
    'Aromatic Amine': 'c[N]',
    'Diamine': '[NH2]-[CH2]-[NH2]',
    'Amino Acid': '[NH2]-[CH]-[C](=[O])[OH]',
    'Glycol': '[C]([OH])[OH]',
    'Epoxide': 'C1OC1',
    'Vinyl Ether': '[C]=[C]-[O]-[C]',
    'Ether Alcohol': '[C]-[O]-[C]-[OH]',
    'Acrylate': '[C]=[C]-[C](=O)[O]',
    
    # Aromatic ring systems
    'Benzene': 'c1ccccc1',
    'Pyridine': 'n1ccccc1',
    'Pyrrole': '[nH]1cccc1',
    'Furan': 'o1cccc1',
    'Thiophene': 's1cccc1',
    'Imidazole': 'c1[nH]cnc1',
    'Thiazole': 'c1ncs1',
    'Xylene': 'c1cc(C)cc(C)c1',
    'Diphenyl': 'c1ccccc1-c2ccccc2',
    
    # Halogen substituents
    'Fluorine': '[F]',
    'Chlorine': '[Cl]',
    'Bromine': '[Br]',
    'Iodine': '[I]',
    'Trifluoromethyl': 'C(F)(F)F',
    'Halogenated Aromatic': 'c1ccccc1-[F,Cl,Br,I]',
    'Vinyl Halide': '[C]=[C]-[F,Cl,Br,I]',
    
    # Sulfur-containing functional groups
    'Sulfonyl': '[S](=O)(=O)',
    'Thionone': '[C]=[S]',
    
    # Alkyl and alkenyl groups
    'Alkyl': '[CH3,CH2,CH]',
    'Alkene': '[C]=[C]',
    'Alkyne': '[C]#[C]',
    'Double Bond (C=C)': '[C]=[C]',
    'Triple Bond (C#C)': '[C]#[C]',
    'Butene': '[C]=[C]-[C]-[C]',
    'Pentene': '[C]=[C]-[C]-[C]-[C]',
    'Allyl': '[C]=[C]-[C]',
    'Vinyl': '[C]=[C]',
    'Phenyl': 'c1ccccc1',
    
    # Miscellaneous groups
    'Silane': '[SiH4]',
    'Alkylamide': '[C]-[N]-[C](=O)',
    'Substituted Alkene': '[C]=[C](-[C])-[C]',
    'Acetate': '[CH3][C](=O)[O-]',
    'Nitrate Ester': '[O]-[N+]([O-])=O',
    'Halohydrin': '[C]([OH])[C][F,Cl,Br,I]',
}