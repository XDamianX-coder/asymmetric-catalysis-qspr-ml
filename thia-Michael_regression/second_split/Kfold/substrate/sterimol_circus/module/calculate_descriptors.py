"""
Module for QSPR analysis with:
>> CircuS/ChyLine descriptors (DOPtools integration)
>> Advanced descriptors (Sterimol, Electronic, ChemBERTa)

Import this module in your QSPR analysis scripts.
Location: module/calculate_descriptors.py
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")


class QSPRConfig:
    """Automatic search for config.yaml."""
    def __init__(self, config_name: str = "config.yaml"):
        self.config_path = self._find_config(config_name)
        self.config = self._load()
        #print(f" Config loaded: {self.config_path.absolute()}")
    
    def _find_config(self, config_name: str) -> Path:
        here = Path(__file__).parent / config_name
        if here.exists(): return here
        qspr_dir = Path(__file__).parent.parent / "QSPR" / config_name  
        if qspr_dir.exists(): return qspr_dir
        cwd = Path.cwd() / config_name
        if cwd.exists(): return cwd
        raise FileNotFoundError(f" config.yaml not found near {__file__}")

    def _load(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        if '.' not in key_path:
            return self.config.get(key_path, default)
        
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value or default

CONFIG = QSPRConfig()


import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
import mordred
from typing import Dict, List, Optional, Tuple
from chython import smiles as parse_smiles

from sklearn.model_selection import  train_test_split

try:
    from morfeus import Sterimol
    MORFEUS_AVAILABLE = True
except ImportError:
    MORFEUS_AVAILABLE = False
    print("WARNING: 'morfeus-ml' not installed. Sterimol values will be NaN.")

import sys
_banner_key = 'DESCRIPTORS_BANNER_'
if os.environ.get(_banner_key) != 'TRUE':
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                calculate_descriptors.py MODULE READY                       ║
║                                                                            ║
║         Import this module in your QSPR analysis scripts                   ║
║         Location: module/calculate_descriptors.py                          ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    os.environ[_banner_key] = 'TRUE'
    sys.modules['descriptors_banner'] = type('Banner', (), {'printed': True})()



# =============================================================================
# MOLECULAR DESCRIPTORS
# =============================================================================

# =============================================================================
# DOPTOOLS CIRCUS & CHYLINE DESCRIPTORS INTEGRATION
# =============================================================================


class CircuSDescriptorCalculator:
    """
    Wrapper for DOPtools CircuS/ChyLine descriptors.
    
    CircuS (Circular Substructures): Fragment descriptors counting circular 
    substructures within certain radius around atoms.
    
    ChyLine (Linear fragments): Descriptors accounting for linear substructures 
    of different sizes.
    
    The fragment structures are kept as columns in the resulting DataFrame in SMILES format.
    
    Reference: POSidorov/DOPtools [https://github.com/POSidorov/DOPtools](https://github.com/POSidorov/DOPtools)
    """
    
    def __init__(self, use_circus: bool = True, use_chyline: bool = True):
        """
        Initialize the descriptor calculator.
        
        Parameters:
        -----------
        use_circus : bool
            Whether to calculate CircuS descriptors (default: True)
        use_chyline : bool
            Whether to calculate ChyLine descriptors (default: True)
        """
        self.use_circus = use_circus
        self.use_chyline = use_chyline
        
        # Initialize descriptor calculators
        if self.use_circus:
            try:
                from doptools.chem.chem_features import ChythonCircus
                self.ChythonCircus = ChythonCircus
                print("OK: CircuS descriptor available")
            except ImportError as e:
                print(f"ISSUE: CircuS not available: {e}")
                self.use_circus = False
        
        if self.use_chyline:
            try:
                from doptools.chem.chem_features import ChythonLinear
                self.ChythonLinear = ChythonLinear
                print("OK: ChyLine descriptor available")
            except ImportError as e:
                print(f"ISSUE: ChyLine not available: {e}")
                self.use_chyline = False
    
    def _parse_smiles(self, smiles_input) -> List:
        """
        Parse SMILES strings to chython molecules.
        
        Parameters:
        -----------
        smiles_input : str, list of str
            Single SMILES or list of SMILES strings
        
        Returns:
        --------
        list
            List of chython molecule objects
        """
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
        else:
            smiles_list = list(smiles_input)
        
        molecules = []
        for smi in smiles_list:
            try:
                mol = parse_smiles(smi)
                molecules.append(mol)
            except Exception as e:
                print(f"Warning: Could not parse SMILES '{smi}': {e}")
                molecules.append(None)
        
        # Filter out None values
        bad_idx = [i for i, m in enumerate(molecules) if m is None]
        if bad_idx:
                raise ValueError(f"Chython parse failed for {len(bad_idx)} SMILES, indexes: {bad_idx}")

        return molecules
    
    def calculate_circus_descriptors(self, 
                                     smiles_input,
                                     lower: int = 1,
                                     upper: int = 3) -> pd.DataFrame:
        """
        Calculate CircuS (Circular Substructures) descriptors.
        
        Parameters:
        -----------
        smiles_input : str or list of str
            SMILES string(s) to process
        lower : int
            Minimum radius for circular fragments (default: 1)
        upper : int
            Maximum radius for circular fragments (default: 3)
        
        Returns:
        --------
        pd.DataFrame
            Feature matrix with fragment SMILES as column names
            Rows represent molecules, columns are fragment counts
        """
        if not self.use_circus:
            print("CircuS descriptor not available")
            return pd.DataFrame()
        
        try:
            # Parse input SMILES
            molecules = self._parse_smiles(smiles_input)
            
            if not molecules:
                print("No valid molecules to process")
                return pd.DataFrame()
            
            # Create CircuS calculator
            calculator = self.ChythonCircus(lower=lower, upper=upper)
            
            # fit_transform returns DataFrame directly
            features_df = calculator.fit_transform(molecules)

            # Add readable prefix with radius info
            prefix = f"CircuS_r{lower}-{upper}_"
            features_df.columns = [prefix + col for col in features_df.columns]
            
            return features_df
        
        except Exception as e:
            print(f"Error calculating CircuS descriptors: {e}")
            return pd.DataFrame()
    
    def calculate_chyline_descriptors(self,
                                     smiles_input,
                                     lower: int = 2,
                                     upper: int = 4) -> pd.DataFrame:
        """
        Calculate ChyLine (Linear fragments) descriptors.
        
        Parameters:
        -----------
        smiles_input : str or list of str
            SMILES string(s) to process
        lower : int
            Minimum length of linear fragments (default: 2)
        upper : int
            Maximum length of linear fragments (default: 4)
        
        Returns:
        --------
        pd.DataFrame
            Feature matrix with fragment SMILES as column names
            Rows represent molecules, columns are fragment counts
        """
        if not self.use_chyline:
            print("ChyLine descriptor not available")
            return pd.DataFrame()
        
        try:
            # Parse input SMILES
            molecules = self._parse_smiles(smiles_input)
            
            if not molecules:
                print("No valid molecules to process")
                return pd.DataFrame()
            
            # Create ChyLine calculator
            calculator = self.ChythonLinear(lower=lower, upper=upper)
            
            # fit_transform returns DataFrame directly
            features_df = calculator.fit_transform(molecules)

            # Add readable prefix with length info
            prefix = f"ChyLine_l{lower}-{upper}_"
            features_df.columns = [prefix + col for col in features_df.columns]
            
            return features_df
        
        except Exception as e:
            print(f"Error calculating ChyLine descriptors: {e}")
            return pd.DataFrame()
    
    def calculate_all(self,
                     smiles_input,
                     circus_params: Optional[Dict] = None,
                     chyline_params: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Calculate both CircuS and ChyLine descriptors.
        
        Parameters:
        -----------
        smiles_input : str or list of str
            SMILES string(s) to process
        circus_params : dict, optional
            Parameters for CircuS: {'lower': int, 'upper': int}
            Default: {'lower': 1, 'upper': 3}
        chyline_params : dict, optional
            Parameters for ChyLine: {'lower': int, 'upper': int}
            Default: {'lower': 2, 'upper': 4}
        
        Returns:
        --------
        tuple:
            (combined_features_df, metadata_dict)
        """
        # Set default parameters
        if circus_params is None:
            circus_params = {'lower': 1, 'upper': 3}
        if chyline_params is None:
            chyline_params = {'lower': 2, 'upper': 4}
        
        metadata = {
            'circus_params': circus_params,
            'chyline_params': chyline_params,
            'circus_features': 0,
            'chyline_features': 0,
            'total_features': 0
        }
        
        all_features = pd.DataFrame()
        
        # Calculate CircuS features
        if self.use_circus:
            try:
                circus_df = self.calculate_circus_descriptors(
                    smiles_input,
                    **circus_params
                )
                if not circus_df.empty:
                    all_features = circus_df
                    metadata['circus_features'] = circus_df.shape[1]
                    print(f" CircuS features: {circus_df.shape}")
            except Exception as e:
                print(f"Error with CircuS calculation: {e}")
        
        # Calculate ChyLine features
        if self.use_chyline:
            try:
                chyline_df = self.calculate_chyline_descriptors(
                    smiles_input,
                    **chyline_params
                )
                if not chyline_df.empty:
                    if all_features.empty:
                        all_features = chyline_df
                    else:
                        # Concatenate horizontally, reset index to align
                        all_features = pd.concat(
                            [all_features, chyline_df],
                            axis=1,
                            ignore_index=False
                        )
                    metadata['chyline_features'] = chyline_df.shape[1]
                    print(f" ChyLine features: {chyline_df.shape}")
            except Exception as e:
                print(f"Error with ChyLine calculation: {e}")
        
        metadata['total_features'] = all_features.shape[1] if not all_features.empty else 0
        
        return all_features, metadata
    
    def calculate_from_dataframe(self,
                                df: pd.DataFrame,
                                smiles_column: str = 'SMILES',
                                circus_params: Optional[Dict] = None,
                                chyline_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate descriptors from a DataFrame with SMILES column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing SMILES column
        smiles_column : str
            Name of the SMILES column (default: 'SMILES')
        circus_params : dict, optional
            Parameters for CircuS
        chyline_params : dict, optional
            Parameters for ChyLine
        
        Returns:
        --------
        pd.DataFrame
            Original DataFrame with descriptor columns appended
        """
        try:
            smiles_list = df[smiles_column].tolist()
            features_df, metadata = self.calculate_all(
                smiles_list,
                circus_params=circus_params,
                chyline_params=chyline_params
            )
            
            if features_df.empty:
                print("No features calculated")
                return df
            
            # Concatenate with original DataFrame
            result = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
            
            print(f"\nFinal shape: {result.shape}")
            print(f"Metadata: {metadata}")
            
            return result
        
        except Exception as e:
            print(f"Error processing DataFrame: {e}")
            return df

# =============================================================================
# SUBSTRUCTURE FINDER FOR STERIMOL & ELECTRONIC DESCRIPTORS
# ============================================================================= 

class UniversalReactiveCCFinder:
    """
    RESPONSIBILITY: 
    1. Identifies the reactive C=C bond associated with an acceptor/functional group.
    2. Supported groups: 
       - Carbonyl (Enone/Aldehyde): C=O
       - Sulfonyl (Vinyl Sulfone): S(=O)2
       - Nitro (Nitroalkene): NO2
       - "Extended" enone: C=C-C-C=O
       - Allylic Alcohol: C(OH)
    3. Defines C_alpha (next to functional group) and C_beta.
    """
    def __init__(self):

        self.patterns = [
            # 1. Enone: C=C-C=O
            # [C:1]=[C:2]-[C:3](=[O:4])
            # 0=Beta, 1=Alpha, 2=Carbonyl, 3=Oxygen
            ('Enone', Chem.MolFromSmarts('[C:1]=[C:2]-[C:3](=[O:4])')),
            
            # 2. Sulfone: C=C-S(=O)=O
            # [C:1]=[C:2]-[S:3](=[O])(=[O])
            # 0=Beta, 1=Alpha, 2=Sulfur
            ('Sulfone', Chem.MolFromSmarts('[C:1]=[C:2]-[S:3](=[O])(=[O])')),
            
            # 3. Nitro: C=C-N(=O)=O
            # [C:1]=[C:2]-[N:3](=[O])[O]  (neutral rep)
            ('Nitro', Chem.MolFromSmarts('[C:1]=[C:2]-[N:3](=[O])[O]')),
            # Charge separated nitro fallback
            ('Nitro', Chem.MolFromSmarts('[C:1]=[C:2]-[N+:3](=[O])[O-]')),
            
            # 4. Extended Enone (Izolowany): C=C-C-C=O
            # Szukamy: C=C połączone z C (sp3), które jest połączone z C=O
            ('Ext-Enone', Chem.MolFromSmarts('[C:1]=[C:2]-[CX4:3]-[C]=[O]')),

            # 5. Allyl Alcohol: C=C-C-OH
            # [C:1]=[C:2]-[C:3][O:4]
            # 0=Beta, 1=Alpha, 2=Carbon_OH, 3=Oxygen_OH
            ('Allyl-OH', Chem.MolFromSmarts('[C:1]=[C:2]-[C:3][O:4]'))
        ]
        
        # Delete empty
        self.patterns = [(name, p) for name, p in self.patterns if p is not None]

    def get_core_map(self, mol):
        match = None
        acc_type = 'Unknown'
        
        # Find first matching
        for name, patt in self.patterns:
            matches = mol.GetSubstructMatches(patt)
            if matches:
                # Bierzemy pierwsze dopasowanie
                match = matches[0]
                acc_type = name
                break
        
        if not match:
            return None
            
        # Map indexes (Beta=0, Alpha=1, Acceptor=2 always in our SMARTS)
        
        core = {
            'beta': match[0],
            'alpha': match[1],
            'acceptor': match[2] # To jest C (enon), S (sulfon), N (nitro), C (alkohol)
        }
        
        scaffold_atoms = set(core.values())

        # Substituents Helper
        def get_subs(core_idx):
            atom = mol.GetAtomWithIdx(core_idx)
            subs = []
            for n in atom.GetNeighbors():
                if n.GetIdx() not in scaffold_atoms:
                    subs.append(n.GetIdx())
            # Sort: Heavy first
            subs.sort(key=lambda x: (-mol.GetAtomWithIdx(x).GetAtomicNum(), -mol.GetAtomWithIdx(x).GetMass()))
            return subs

        # Build map
        # Changing keys to small letters in core_atoms to fit SterimolCalculator
        # 'c_alpha', 'c_beta', 'acceptor'
        
        core_map = {
            'core_atoms': {
                'c_beta': core['beta'],
                'c_alpha': core['alpha'],
                'acceptor': core['acceptor']
            },
            'subs': {
                'Beta': get_subs(core['beta']),
                'Alpha': get_subs(core['alpha'])
                # We do not collect substituents for acceptor as we are calculating Sterimol only for C=C
            },
            'acceptor_type': acc_type
        }
        
        return core_map
    


def get_thiol_core_map(mol):
    """
    Generates a core_map for a thiol compatible with SterimolCalculator.
    Uses the safe pattern '[#16;H1]' (sulfur with one hydrogen)
    to avoid confusion with thioethers or sulfones.
    """
    # 1. Find the -SH group (Sulfur with 1 hydrogen)
    # This is a "future-proof" safe approach
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[#16;H1]'))
    
    if not matches:
        return None
    
    # Take the first -SH group found (usually there is only one)
    s_idx = matches[0][0] 
    s_atom = mol.GetAtomWithIdx(s_idx)
    
    # 2. Find the carbon attached to the sulfur (Anchor for the rest of the molecule)
    c_neighbor = None
    for n in s_atom.GetNeighbors():
        # Look for a heavy neighbor (not hydrogen)
        if n.GetAtomicNum() != 1: 
            c_neighbor = n
            break
            
    if c_neighbor is None: # E.g., in the case of H2S (only hydrogens)
        return None

    c_idx = c_neighbor.GetIdx()
    
    # 3. Identify substituents on the carbon
    # This defines the "Rest" (R) in the R-SH structure
    subs_indices = []
    for n in c_neighbor.GetNeighbors():
        if n.GetIdx() != s_idx: # Everything that isn't the sulfur
            subs_indices.append(n.GetIdx())
    
    # Sort substituents (for reproducibility of R1, R2...)
    subs_indices.sort(key=lambda x: (-mol.GetAtomWithIdx(x).GetAtomicNum(), -mol.GetAtomWithIdx(x).GetMass()))

    # 4. Mapping to the "Pseudo-Enone" structure for SterimolCalculator
    # C_alpha = Sulfur (Where we stand)
    # C_beta  = Carbon (The direction we look)
    # Subs    = What hangs off the carbon (What we measure)
    
    core_map = {
        'core_atoms': {
            'c_alpha': s_idx,   # Sulfur (Anchor)
            'c_beta': c_idx,    # Carbon (Base Axis)
            'acceptor': s_idx   # Point for retrieving charge (Head Charge)
        },
        'subs': {
            'Alpha': subs_indices, # Substituents "seen" from the sulfur side
        },
        'acceptor_type': 'Thiol'
    }
    
    return core_map

# =============================================================================
# STERIMOL DESCRIPTORS L, B1, B5 CALCULATOR
# =============================================================================

class SterimolCalculator:
    """
    Calculate Sterimol (L, B1, B5) relative to the C=C bond axis using BFS extraction.
    Maintains compatibility with original code structure.
    """

    def calculate(self, mol, core_map):
        results = {}
        if not core_map:
            return results

        if not MORFEUS_AVAILABLE: # Ensure this global flag exists or remove check
             print("Morfeus not available")
             return results

        # Map 'pos_name' to the corresponding core atom index
        # For 'Alpha' subs -> anchor is c_alpha
        # For 'Beta' subs -> anchor is c_beta
        anchor_map = {
            'Alpha': core_map['core_atoms']['c_alpha'],
            'Beta': core_map['core_atoms']['c_beta']
        }
        
        # We also need the "other" C=C atom to define the axis/blocking
        axis_base_map = {
            'Alpha': core_map['core_atoms']['c_beta'],  # Axis for Alpha subs comes from Beta
            'Beta': core_map['core_atoms']['c_alpha']   # Axis for Beta subs comes from Alpha
        }

        for pos_name, subs in core_map['subs'].items():
            anchor_idx = anchor_map[pos_name]
            base_idx = axis_base_map[pos_name] # The other end of C=C

            # R1 (always first substituent)
            if len(subs) > 0:
                self._calc_one(mol, subs[0], anchor_idx, base_idx, f"{pos_name}_R1", results)

            # R2 (second substituent, typically on Beta)
            if len(subs) > 1:
                self._calc_one(mol, subs[1], anchor_idx, base_idx, f"{pos_name}_R2", results)

        return results

    def _calc_one(self, mol, sub_idx, anchor_idx, base_idx, prefix, results):
        """
        Calculates Sterimol for substituent starting at sub_idx attached to anchor_idx.
        base_idx is the other end of the double bond (used for axis definition).
        """
        try:
            # 1. Identify substituent atoms (BFS) - EXACTLY AS BEFORE
            # But we MUST block base_idx too, otherwise BFS might go around rings back to C=C
            sub_atoms = set([sub_idx])
            stack = [sub_idx]
            visited = {sub_idx, anchor_idx, base_idx}  # Block anchor AND base (C=C atoms)

            while stack:
                curr = stack.pop()
                atom = mol.GetAtomWithIdx(curr)
                for n in atom.GetNeighbors():
                    n_idx = n.GetIdx()
                    if n_idx not in visited:
                        visited.add(n_idx)
                        sub_atoms.add(n_idx)
                        stack.append(n_idx)

            # Add anchor_idx and base_idx to relevant indices
            # We include base_idx to define the axis direction C_base -> C_anchor
            relevant_indices = list(sub_atoms) + [anchor_idx, base_idx]

            # Get elements and coords ONLY for these atoms
            elements = []
            coordinates = []
            conf = mol.GetConformer()

            # Map old_index -> new_index (1-based for Morfeus)
            idx_map = {old_idx: i+1 for i, old_idx in enumerate(relevant_indices)}

            for idx in relevant_indices:
                atom = mol.GetAtomWithIdx(idx)
                elements.append(atom.GetSymbol())
                pos = conf.GetAtomPosition(idx)
                coordinates.append([pos.x, pos.y, pos.z])

            # New indices for Sterimol
            new_sub_idx = idx_map[sub_idx]
            new_anchor_idx = idx_map[anchor_idx]
            new_base_idx = idx_map[base_idx]
            
            # axis = C=C bond (seems better for Michael acceptors):
            st = Sterimol(elements, coordinates, new_base_idx, new_anchor_idx)

            results[f"Sterimol_{prefix}_L"] = st.L_value
            results[f"Sterimol_{prefix}_B1"] = st.B_1_value
            results[f"Sterimol_{prefix}_B5"] = st.B_5_value

        except Exception as e:
            print(f"Error {prefix}: {e}")
            results[f"Sterimol_{prefix}_L"] = np.nan
            results[f"Sterimol_{prefix}_B1"] = np.nan
            results[f"Sterimol_{prefix}_B5"] = np.nan

# =============================================================================
# ELECTRONIC DESCRIPTOR CALCULATOR
# =============================================================================

class ElectronicCalculator:
    """
    Calculate Electronic props (Gasteiger Charges, Differentials, and Substituent effects).
    Updated to work with UniversalReactiveCCFinder core_map.
    """

    pauling_electronegativity = {
        1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
        15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    }

    def calculate(self, mol, core_map):
        results = {}
        if not core_map:
            return results

        # CORE ATOMS (C_alpha, C_beta, Acceptor)
        # Using .get() allows flexibility if 'acceptor' key name changes slightly
        # UniversalFinder uses: 'c_alpha', 'c_beta', 'acceptor'
        core_keys_map = {
            'c_alpha': 'Alpha',
            'c_beta': 'Beta',
            'acceptor': 'Acceptor' # Generalized name (Carbonyl/Sulfonyl/Nitro head)
        }

        core_charges = {}
        for core_key, res_prefix in core_keys_map.items():
            idx = core_map['core_atoms'].get(core_key)
            if idx is not None:
                atom = mol.GetAtomWithIdx(idx)
                try:
                    val = float(atom.GetProp('_GasteigerCharge'))
                except:
                    val = np.nan
                results[f'{res_prefix}_Charge'] = val
                core_charges[res_prefix] = val

        # SUBSTITUENTS (Dynamic Props: Charge Sum of first 3 atoms)
        # UniversalFinder returns subs for 'Alpha' and 'Beta'
        for pos_name, subs in core_map['subs'].items():
            # R1 (First substituent)
            if len(subs) > 0:
                self._get_sub_props_extended(mol, subs[0], f"{pos_name}_R1", results)

            # R2 (Second substituent)
            if len(subs) > 1:
                self._get_sub_props_extended(mol, subs[1], f"{pos_name}_R2", results)

        # KEY DIFFERENCES (Reactivity Indices)
        try:
            # Polarization of the C=C bond (Push-Pull effect)
            if 'Beta' in core_charges and 'Alpha' in core_charges:
                results['Diff_Charge_Beta_Alpha'] = core_charges['Beta'] - core_charges['Alpha']
            
            # Polarization between Acceptor and Alpha (Inductive pull)
            if 'Acceptor' in core_charges and 'Alpha' in core_charges:
                results['Diff_Charge_Acceptor_Alpha'] = core_charges['Acceptor'] - core_charges['Alpha']
                
        except Exception:
            pass

        return results

    def _get_sub_props_extended(self, mol, start_atom_idx, prefix, results_dict):
        """
        Calculates properties for a substituent:
        1. EN of the first atom (static).
        2. Gasteiger Charge of the first atom.
        3. Sum of Gasteiger Charges for the first 3 atoms (local polarity).
        """
        try:
            # BFS to get first 3 atoms of the substituent
            sub_atoms_depth_3 = []
            queue = [(start_atom_idx, 0)] # (idx, depth)
            visited = {start_atom_idx}
            
            # Simple traversal (ignoring blocking core since we start OUTSIDE core)
            # Just getting first 3 unique atoms found
            
            idx = 0
            while idx < len(queue) and len(queue) < 3:
                curr, depth = queue[idx]
                idx += 1
                
                atom = mol.GetAtomWithIdx(curr)
                # Add neighbors to queue
                if depth < 2: # Don't go deeper than needed for 3 atoms
                    for n in atom.GetNeighbors():
                        if n.GetIdx() not in visited:
                            visited.add(n.GetIdx())
                            queue.append((n.GetIdx(), depth + 1))
                            if len(queue) >= 3: break
            
            # Collect atoms from queue (up to 3)
            collected_indices = [x[0] for x in queue]
            
            # First Atom Props
            first_atom = mol.GetAtomWithIdx(start_atom_idx)
            results_dict[f'{prefix}_EN'] = self.pauling_electronegativity.get(first_atom.GetAtomicNum(), np.nan)
            
            try:
                first_charge = float(first_atom.GetProp('_GasteigerCharge'))
            except:
                first_charge = np.nan
            results_dict[f'{prefix}_Charge_Head'] = first_charge

            # Sum of Charges (First 3 atoms)
            sum_charge = 0.0
            count = 0
            for idx in collected_indices:
                try:
                    c = float(mol.GetAtomWithIdx(idx).GetProp('_GasteigerCharge'))
                    sum_charge += c
                    count += 1
                except:
                    pass
            
            if count > 0:
                results_dict[f'{prefix}_Charge_Sum3'] = sum_charge
            else:
                results_dict[f'{prefix}_Charge_Sum3'] = np.nan

        except Exception as e:
            print(f"Error {prefix}: {e}")
            results_dict[f'{prefix}_EN'] = np.nan
            results_dict[f'{prefix}_Charge_Head'] = np.nan
            results_dict[f'{prefix}_Charge_Sum3'] = np.nan

# =============================================================================
# CHEMBERTA EMBEDDINGS (DESCRIPTORS) CALCULATOR
# =============================================================================

class ChemBERTaEmbedder:
    """
    Calculates molecular embeddings using pre-trained ChemBERTa transformers.
    Uses Mean Pooling of the last hidden state for better molecule representation.
    """

    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MTR"):
        self.model_name = model_name
        self.hidden_size = 384  # Default for 77M-MTR, will be updated automatically
        
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
            
            # Detect hidden size from config to be safe
            if hasattr(self.model.config, 'hidden_size'):
                self.hidden_size = self.model.config.hidden_size
                print(f"ChemBERTa model output size: {self.hidden_size}")
                
            self.available = True
        except ImportError:
            print("Warning: 'transformers' library not found. ChemBERTa skipped.")
            self.available = False
        except Exception as e:
            print(f"Warning: Could not load ChemBERTa model: {e}")
            self.available = False

    def get_embedding(self, smiles):
        """
        Returns a numpy array representing the molecule embedding.
        """
        if not self.available:
            return np.zeros(self.hidden_size)

        try:
            import torch
            # Tokenize SMILES
            inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get last hidden states: shape (Batch=1, Seq_Len, Hidden_Size)
                token_embeddings = outputs.last_hidden_state[0] 
                
                # MEAN POOLING STRATEGY
                # Calculate average of all token embeddings (better than CLS for MLM models)
                # dim=0 means average across sequence length
                embedding = torch.mean(token_embeddings, dim=0).numpy()
                
                # ALTERNATIVE: CLS Token Strategy (strict BERT style)
                # embedding = token_embeddings[0].numpy()
                
            return embedding
            
        except Exception as e:
            print(f"Error ChemBERTa embedding SMILES {smiles}: {e}")
            return np.zeros(self.hidden_size)
        
        
# =============================================================================
# SMART ROOTING UTILS
# =============================================================================

class SmartRooter:
    """ Ensures SMILES rooted at the Reactive C_alpha to ensure consistent folding. 
    - For Acceptors (Chalkones): Roots at Alpha-Carbon (via UniversalReactiveCCFinder).
    - For Thiols: Roots at the Sulfur atom of the SH group.
    """
    def __init__(self):
        self.finder = UniversalReactiveCCFinder()

        self.thiol_pattern = Chem.MolFromSmarts('[#16;H1]') 

    def get_root_atom_idx(self, mol, expected_type='auto'):
        """
        Determines the best atom index to start the SMILES string.
        Strategy:
        1. Check if it's a Michael Acceptor (using UniversalFinder).
        2. If not, check if it's a Thiol (using SMARTS).
        3. Fallback to -1 (random/default).
        """
        
        # --- PATH 1: THIOL (Explicit Request) ---
        if expected_type == 'thiol':
            # Find the single Sulfur atom
            matches = mol.GetSubstructMatches(self.thiol_pattern)
            if matches:
                # matches[0][0] is the index of the first atom of the first match
                return matches[0][0]
            else:
                print(f"Warning: No Sulfur found in expected Thiol molecule!")
                return -1

        # --- PATH 2: ACCEPTOR / CHALKONE (Explicit Request) ---
        if expected_type == 'acceptor':
            core_map = self.finder.get_core_map(mol)
            if core_map:
                return core_map['core_atoms']['c_alpha']
            else:
                # Optional: Log warning if acceptor pattern is missing in a chalkone column
                # print(f"Warning: No acceptor pattern found in expected Chalkone!")
                return -1

        # --- PATH 3: AUTO (Fallback / Hybrid) ---
        # Try Acceptor first (more specific structure)
        core_map = self.finder.get_core_map(mol)
        if core_map:
            return core_map['core_atoms']['c_alpha']
            
        # Try Sulfur second
        matches = mol.GetSubstructMatches(self.thiol_pattern)
        if matches:
            return matches[0][0]

        return -1


# Initialize globally
smart_rooter = SmartRooter()


def get_standardized_mol(smiles, molecule_type='auto'):
    """
    Returns (smart_rooted_smiles, mol_object_with_Hs)
    molecule_type: 'thiol' (roots at S) or 'acceptor' (roots at alpha-C)
    """

    try:
        if pd.isna(smiles):
            return None, None

        # Base Molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            return None, None

        # Find rooting point
        root_idx = smart_rooter.get_root_atom_idx(mol, expected_type=molecule_type)

        # Generate SMILES
        if root_idx != -1:
            smart_smiles = Chem.MolToSmiles(mol, rootedAtAtom=root_idx, canonical=True, isomericSmiles=True)
        else:
            print(f" ROOTING WARNING: No rooting core found for SMILES: {smiles}")
            print(f" -> Using default RDKit canonicalization (random start atom).")
            smart_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        # 3D Conformer with Hs (Deterministic)
        mol_3d = Chem.MolFromSmiles(smart_smiles)
        mol_3d = Chem.AddHs(mol_3d)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42

        res = AllChem.EmbedMolecule(mol_3d, params)
        if res == -1:
            print(f"Warning: 3D embedding failed for {smiles}")
            return None, None

        try:
            val = AllChem.MMFFOptimizeMolecule(mol_3d)
            if val == -1:
                print(f"Warning: MMFF optimization did not converge for {smiles}")

        except Exception as e:
            print(f"Warning: MMFF optimization crashed for {smiles}: {e}")


        # Compute Gasteiger Charges
        try:
            AllChem.ComputeGasteigerCharges(mol_3d)
        except:
            print(f"Warning: Could not compute Gasteiger charges for {smiles}")

        return smart_smiles, mol_3d

    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None, None

# =============================================================================
# PREPARE FINAL DATAFRAME WITH DESCRIPTORS
# =============================================================================

TARGET = 'ee'

def prepare_data(data: str, target_col: str, name: str, mode: str,
                 output_dir: Optional[str] = None):
    """
    Load Excel data and calculate molecular descriptors.

    Parameters
    ----------
    data : str
        Path to Excel file with SMILES
    target_col : str
        Target property column name
    name : str
        Output filename prefix
    mode : str
        'models_creation' (auto train/test split) or other
    output_dir : Optional[str]
        Directory to save descriptor files

    Returns
    -------
    For mode='models_creation': (train_df, test_df)
    Otherwise: full_df with descriptors + target
    """
    # Config-driven descriptors
    desc_config = CONFIG.get('descriptors')
    mordred_features = desc_config.get('mordred', True)
    fingerprints = desc_config.get('morgan_fp', True)
    sterimol_electronic = desc_config.get('sterimol_electronic', True)
    chembert = desc_config.get('chemberta', True)

    circus_config = desc_config.get('circus', {})
    include_circus = circus_config.get('enabled', False)
    circus_params = {'lower': circus_config.get('lower_radius', 1),
                     'upper': circus_config.get('upper_radius', 3)} if include_circus else None
    
    chyline_config = desc_config.get('chyline', {})
    include_chyline = chyline_config.get('enabled', False)
    chyline_params = {'lower': chyline_config.get('lower_length', 2),
                      'upper': chyline_config.get('upper_length', 4)} if include_chyline else None

    print("\n" + "="*75)
    print("LOADING DATA FOR UNIFIED DESCRIPTOR CALCULATION")
    print("="*75)

    # Load data
    df = pd.read_excel(data)
    targets = df[target_col].values
    indices = range(len(df))

    print("  DESCRIPTOR PIPELINE:")
    print(f" Mordred: {mordred_features}")
    print(f" Morgan FP: {fingerprints}")
    print(f" Sterimol + Electronic: {sterimol_electronic}")
    print(f" ChemBERTa: {chembert}")
    print(f" CircuS: {include_circus}")
    print(f" ChyLine: {include_chyline}")

    # Create output directory
    output_dir = output_dir or CONFIG.get('paths.descriptors')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate/Clean descriptors
    print(f"\n" + "-"*75)
    print("Calculating/Cleaning descriptors ...")
    print("-"*75)

    molecular_descriptors_cleaned, canonical_smiles_list = _calculate_and_clean_descriptors(
        df, output_path, name, mordred_features, fingerprints, sterimol_electronic,
        chembert, include_circus, include_chyline, circus_params, chyline_params, targets, target_col, mode
    )

    filename = f'initial_vs_new_SMILES_{name}_.xlsx'
    comparison_file_path = output_path / filename
    # Mode-dependent split + SMILES export
    if mode == 'models_creation':
        num_bins = 5
        df['ee_bin'] = pd.cut(df['ee'], bins=num_bins, labels=False)

        train_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=df['ee_bin'], random_state=42)
        train_df = molecular_descriptors_cleaned.loc[train_idx].copy()
        test_df = molecular_descriptors_cleaned.loc[test_idx].copy()
        
        print(f"Internal split: train={len(train_idx)}, test={len(test_idx)}")
        print("Returning: train_df, test_df")

        if not comparison_file_path.exists():
            print("Saving SMILES comparison...")
            combined_df = pd.DataFrame(data=df['SMILES'], columns=['SMILES'])
            combined_df['dataset'] = None
            combined_df.loc[train_idx, 'dataset'] = 'train'
            combined_df.loc[test_idx, 'dataset'] = 'test'
            combined_df[target_col] = targets

            _save_smiles_comparison(combined_df, canonical_smiles_list, output_path, name, mode)
        else:
            pass
        return train_df, test_df
    else:
        print("Returning: full descriptor set")
        
        full_df = molecular_descriptors_cleaned.copy()
        combined_df = pd.DataFrame(data=df, columns=['SMILES'])
        combined_df[target_col] = targets
        if not comparison_file_path.exists():
            print("Saving SMILES comparison...")
            _save_smiles_comparison(combined_df, canonical_smiles_list, output_path, name, mode)
        else:
            pass
        
        return full_df


def _calculate_and_clean_descriptors(df, output_path, name, mordred_features, fingerprints, 
                                   sterimol_electronic, chembert, include_circus, include_chyline,
                                   circus_params, chyline_params, targets, target_col, mode):
    """Calculate descriptors with cache + NaN cleaning."""
    cleaned_path = output_path / f'molecular_descriptors_cleaned_{name}_.parquet'
    raw_path = output_path / f'molecular_descriptors_raw_{name}_.parquet'
    
    try:
        molecular_descriptors_cleaned = pd.read_parquet(cleaned_path)
        if "Unnamed: 0" in molecular_descriptors_cleaned.columns:
            molecular_descriptors_cleaned = molecular_descriptors_cleaned.drop(columns=["Unnamed: 0"])
        print(f"SUCCESS: Loaded CLEANED descriptors from {cleaned_path}")
        print(f"Shape: {molecular_descriptors_cleaned.shape}")
        return molecular_descriptors_cleaned, None

    except FileNotFoundError:
        # CALCULATE FROM SCRATCH
        print("No cache found. Calculating descriptors from scratch...")
        print("\n" + "-"*75)
        print("Canonicalizing SMILES & Generating 3D Conformers")
        print("-"*75)

        smiles_dot_list = []  # Circus, Fingerprints, ChemBERTa
        mols_tiol_list = []   # Mordred/Electronic/Sterimol thiol
        mols_chalkon_list = [] # Mordred/Electronic/Sterimol chalcone

        print("Processing Tiol + Chalkon pairs (this may take a moment)...")
        for idx, row in df.iterrows():
            # TIOL (Rooting on SH)
            smi_t = row.get('SMILES_tiol')
            can_t, mol_t = get_standardized_mol(smi_t, molecule_type='thiol')
            
            # CHALKON (Rooting on Enone)
            smi_c = row.get('SMILES_chalkon')
            can_c, mol_c = get_standardized_mol(smi_c, molecule_type='acceptor')

            # Issues managing
            if mol_t is None or mol_c is None:
                smiles_dot_list.append(None)
                mols_tiol_list.append(None)
                mols_chalkon_list.append(None)
            else:
                # DOT (Nucleophile.Electrophile)
                smiles_dot_list.append(f"{can_t}.{can_c}")
                mols_tiol_list.append(mol_t)
                mols_chalkon_list.append(mol_c)

        molecular_descriptors = pd.DataFrame(index=range(len(smiles_dot_list)))
        # Mordred
        if mordred_features:
            mol_dot = [Chem.MolFromSmiles(smi) for smi in smiles_dot_list]
            print(" >> Calculating Mordred descriptors...")
            calc = Calculator(descriptors, ignore_3D=True)
            mord_df = calc.pandas(mol_dot)
            mord_df = mord_df.applymap(
                lambda x: np.nan if isinstance(x, (mordred.error.Missing, mordred.error.Error)) else x)
            mord_df = mord_df[sorted(mord_df.columns)]
            molecular_descriptors = pd.concat([molecular_descriptors, mord_df], axis=1)
            print(f" OK: Mordred: {mord_df.shape}")

        # Morgan fingerprints
        if fingerprints:
            print(" >> Calculating Morgan fingerprints...")
            mols_dot_list = [Chem.MolFromSmiles(s) if s else None for s in smiles_dot_list]
            fingerprints_counts = []
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
            for mol in mols_dot_list:
                if mol is not None:
                    fp = AllChem.GetMorganFingerprint(mol, radius=2, useCounts=True)
                    count_dict = fp.GetNonzeroElements()
                    count_array = np.zeros(2048)
                    for key, val in count_dict.items():
                        count_array[key % 2048] = val
                    fingerprints_counts.append(count_array)
                else:
                    fingerprints_counts.append(np.nan * np.zeros(2048))
            counts_df = pd.DataFrame(fingerprints_counts, columns=[f'C_FP_{i}' for i in range(2048)])
            counts_df.index = molecular_descriptors.index
            molecular_descriptors = pd.concat([molecular_descriptors, counts_df], axis=1)
            print(f" OK: Morgan FP: {counts_df.shape}")

        # Sterimol + Electronic
        if sterimol_electronic:
            print(" >> Calculating Reactive (Tiol & Chalkon) Sterimol & Electronic descriptors...")

            finder = UniversalReactiveCCFinder()
            sterimol_calc = SterimolCalculator()
            electronic_calc = ElectronicCalculator()

            # --- CHALCONE ---
            print("    ... calculating Chalcone Sterimol/Electronic")
            desc_chalkon = []
            for mol in mols_chalkon_list:
                desc = {}
                if mol:
                    core_map = finder.get_core_map(mol)
                    if core_map:
                         desc.update(sterimol_calc.calculate(mol, core_map))
                         desc.update(electronic_calc.calculate(mol, core_map))
                desc_chalkon.append(desc)

            df_st_c = pd.DataFrame(desc_chalkon).add_suffix('_Chalcone')

            # --- THIOL ---
            print("    ... calculating Thiol Sterimol/Electronic")
            desc_tiol = []
            for mol in mols_tiol_list:
                 d = {}
                 if mol:
                     core_map_t = get_thiol_core_map(mol)
                     if core_map_t:
                         d.update(sterimol_calc.calculate(mol, core_map_t))
                         d.update(electronic_calc.calculate(mol, core_map_t))
                 desc_tiol.append(d)
             
            df_st_t = pd.DataFrame(desc_tiol).add_suffix('_Thiol')

            try:
                # Merge into main DF
                st_df = pd.concat([df_st_c, df_st_t], axis=1)
                st_df.index = molecular_descriptors.index
                molecular_descriptors = pd.concat([molecular_descriptors, st_df], axis=1)
                print(f" OK: Added {st_df.shape[1]} C=C features")
            except Exception as e:
                print(f" Sterimol + Electronic error: {e}")

        # ---------------------------------------------------------
        # NEW: INTERACTION FEATURES (Sterimol Interactions ONLY)
        # ---------------------------------------------------------
        if sterimol_electronic:
            print(" >> Generating Interaction Features (Sterimol/Electronic only)...")
            try:
                X_int = build_reactive_interactions(
                    molecular_descriptors,
                    make_cross_reactant=True,
                    make_within_reactant_asymmetry=True
                )
                X_int.index = molecular_descriptors.index # Ensure index matches original rows
                if not molecular_descriptors.empty:
                    molecular_descriptors.index = df_st_t.index
                molecular_descriptors = pd.concat([molecular_descriptors, X_int], axis=1)
                print(f" OK: Added {X_int.shape[1]} pure reactive interaction features.")
                
            except Exception as e:
                print(f" Warning: Could not build interactions: {e}")

        # ChemBERTa
        if chembert:
            print(">> Calculating ChemBERTa embeddings...")
            embedder = ChemBERTaEmbedder()
            if embedder.available:
                embs = [embedder.get_embedding(smi) for smi in smiles_dot_list]
                cb_df = pd.DataFrame(embs, columns=[f'ChemBERTa_{i}' for i in range(384)])
                cb_df.index = molecular_descriptors.index
                molecular_descriptors = pd.concat([molecular_descriptors, cb_df], axis=1)
                print(f" OK: ChemBERTa: {cb_df.shape}")

        # CircuS and ChyLine
        if include_circus or include_chyline:
            print(">> Calculating CircuS/ChyLine descriptors (Tiol.Chalkon pair)...")
            circus_calc = CircuSDescriptorCalculator(
                use_circus=include_circus, use_chyline=include_chyline
            )
            features_df, metadata = circus_calc.calculate_all(
                smiles_dot_list, circus_params=circus_params, chyline_params=chyline_params
            )
            if not features_df.empty:
                print(f"OK: Generated {metadata['total_features']} fragment features")
                print(f" - CircuS: {metadata['circus_features']}")
                print(f" - ChyLine: {metadata['chyline_features']}")
                features_df.index = molecular_descriptors.index
                molecular_descriptors = pd.concat([molecular_descriptors, features_df], axis=1, sort=False)
            else:
                print("ISSUE: No CircuS/ChyLine features generated")

        print(f"Raw descriptors shape: {molecular_descriptors.shape}")
        # SAVE RAW for future use
        molecular_descriptors.to_parquet(raw_path, index=False)
        print(f"Raw descriptors cached: {raw_path}")

    if mode == 'models_creation':# drop all NaNs
        print("\n >> CLEANING DESCRIPTORS (drop all NaNs)...")
        cols_before = molecular_descriptors.shape[1]
        
        # Drop columns with any NaN (strict for small dataset)
        molecular_descriptors_cleaned = molecular_descriptors.dropna(axis=1, how='any')
        cols_after = molecular_descriptors_cleaned.shape[1]
        print(f" >> Dropped {cols_before - cols_after} empty columns")
        # Add target (match length)
        molecular_descriptors_cleaned[target_col] = targets[:len(molecular_descriptors_cleaned)]
    else:
        print("No NaN cleaning was performed...")
        # Add target (match length)
        molecular_descriptors_cleaned = molecular_descriptors
        molecular_descriptors_cleaned[target_col] = targets[:len(molecular_descriptors_cleaned)]

    # SAVE CLEANED for fastest future loads
    molecular_descriptors_cleaned.to_parquet(cleaned_path, index=False)
    print(f"OK: descriptors cached: {cleaned_path}")

    # Summary
    print(f"\n" + "="*75)
    print("DESCRIPTOR CALCULATION SUMMARY")
    print("="*75)
    print(f"Final shape: {molecular_descriptors_cleaned.shape[0]} samples × {molecular_descriptors_cleaned.shape[1]-1} features")
    print(f"Target: {target_col}")
    print(f"Output dir: {output_path}")
    print("="*75)

    return molecular_descriptors_cleaned.reset_index(drop=True), smiles_dot_list


def _save_smiles_comparison(df_with_smiles, canonical_smiles_list, output_path, name, mode):
    """Save initial_vs_new_SMILES.xlsx """
    data_smiles = pd.DataFrame(index=df_with_smiles.index)
    data_smiles['Initial_SMILES'] = df_with_smiles['SMILES']
    all_new_smiles = pd.Series(canonical_smiles_list, index=df_with_smiles.index)
    data_smiles['New_SMILES'] = all_new_smiles.loc[df_with_smiles.index]

    data_smiles[TARGET] = df_with_smiles[TARGET]
    
    if mode == 'models_creation':
        data_smiles['dataset'] = df_with_smiles['dataset']
    
    filename = f'initial_vs_new_SMILES_{name}_.xlsx'
    data_smiles.to_excel(output_path / filename, index=True)
    print(f"OK: SMILES comparison saved: {output_path / filename}")


import re
def build_reactive_interactions(
    X,
    make_cross_reactant=True,          # Tiol * Chalcone interaction
    make_within_reactant_asymmetry=True # |R1 - R2| asymmetry
):
    """
    Builds interaction features ONLY for Reactive Descriptors (Sterimol & Electronic).
    Assumes Mordred is already calculated for the pair and should be left alone.
    """
    X = X.copy()

    # 1. IDENTIFY REACTIVE COLUMNS
    # ----------------------------
    # Define regex patterns for your specific naming convention
    # Matches: Sterimol_Beta_R1_L_Chalcone, Sterimol_Beta_R5_L_Chalcone, etc.
    chalcone_pat = re.compile(r"(_Chalcone|Chalcone)$")
    thiol_pat    = re.compile(r"(_Thiol|Thiol)$")
    
    sterimol_kw   = "Sterimol"
    electronic_kw = r"(Charge|EN|Diff)"  # Keywords for electronic props

    # Select all columns
    all_cols = X.columns.tolist()

    # Filter columns belonging to Chalcone vs Thiol
    c_cols = [c for c in all_cols if chalcone_pat.search(c)]
    t_cols = [c for c in all_cols if thiol_pat.search(c)]

    # Further split into Sterimol vs Electronic
    ster_C = [c for c in c_cols if sterimol_kw in c]
    ster_T = [c for c in t_cols if sterimol_kw in c]

    elec_C = [c for c in c_cols if re.search(electronic_kw, c)]
    elec_T = [c for c in t_cols if re.search(electronic_kw, c)]

    frames = []

    # 2. CROSS-REACTANT INTERACTIONS (Key for QSPR)
    # ---------------------------------------------
    # These capture the "Fit" between Nucleophile and Electrophile
    if make_cross_reactant:
        # A. Steric Fit: Sterimol * Sterimol
        # e.g., Thiol_Width * Chalcone_Width (Clash potential)
        frames.append(make_interactions(X, ster_T, ster_C, op="mul", prefix="X_STER"))

        # B. Electronic Match: Electronic * Electronic
        # e.g., Thiol_Charge * Chalcone_Charge (Electrostatic interaction)
        frames.append(make_interactions(X, elec_T, elec_C, op="mul", prefix="X_ELEC"))

        # C. Mixed: Steric * Electronic (Secondary effects)
        # e.g., Does a bulky group (Sterimol) shield a charge (Electronic)?
        frames.append(make_interactions(X, ster_T, elec_C, op="mul", prefix="X_STER_T_x_ELEC_C"))
        frames.append(make_interactions(X, elec_T, ster_C, op="mul", prefix="X_ELEC_T_x_STER_C"))

    # 3. WITHIN-REACTANT ASYMMETRY (Key for Enantioselectivity)
    # ---------------------------------------------------------
    # These capture "How different is the left side from the right side?"
    if make_within_reactant_asymmetry:
        # Chalcone Asymmetry (e.g. |R1_L - R2_L|)
        frames.append(r1_r2_absdiff(X, ster_C, tag="_C"))
        frames.append(r1_r2_absdiff(X, elec_C, tag="_C"))

        # Thiol Asymmetry
        frames.append(r1_r2_absdiff(X, ster_T, tag="_T")) 
        frames.append(r1_r2_absdiff(X, elec_T, tag="_T"))

    if not frames:
        print("Warning: No reactive columns found to interact.")
        return pd.DataFrame(index=X.index)

    # Concatenate all new features
    X_int = pd.concat(frames, axis=1)

    # Cleanup: Remove columns that are all NaN or Duplicates
    X_int = X_int.dropna(axis=1, how="all")
    X_int = X_int.loc[:, ~X_int.T.duplicated()]

    return X_int

# --- Helper functions required by the above ---

def make_interactions(df, cols_a, cols_b, op="mul", prefix="INT"):
    """Helper to multiply/combine two lists of columns."""
    out = {}
    # Use numpy arrays for speed
    if not cols_a or not cols_b:
        return pd.DataFrame()
        
    df_a = df[cols_a].values
    df_b = df[cols_b].values
    
    # Simple loop is safe enough for <1000 features
    for i, col_a in enumerate(cols_a):
        for j, col_b in enumerate(cols_b):
            name = f"{prefix}__{col_a}_x_{col_b}"
            # Shorten names if they are too long (optional)
            name = name.replace("Sterimol", "St").replace("Chalcone", "C").replace("Thiol", "T")
            
            if op == "mul":
                out[name] = df[col_a] * df[col_b]
            # Add other ops if needed
            
    return pd.DataFrame(out, index=df.index)

def r1_r2_absdiff(df, cols, tag=""):
    """Helper to calculate |R1 - R2| asymmetry."""
    out = {}
    processed = set()
    
    for c1 in cols:
        if "R1" in c1:
            # Try to find matching R2
            c2 = c1.replace("R1", "R2")
            if c2 in df.columns:
                # Short name
                base_name = c1.replace("R1", "").replace("Chalcone", "").replace("Thiol", "").replace("_", "")
                name = f"ASYM{tag}_{base_name}_R1_vs_R2"
                
                out[name] = (df[c1] - df[c2]).abs()
    
    return pd.DataFrame(out, index=df.index)