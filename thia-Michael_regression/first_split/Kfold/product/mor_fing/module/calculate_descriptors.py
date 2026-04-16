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

class ProductCSBondFinder:
    """
    RESPONSIBILITY:
    1. Identifies the newly formed C-S bond in the Michael Addition Product.
    2. Motif: Carbonyl-C(alpha)-C(beta)-S(thiol).
    3. Defines the bond axis C_beta - S_thiol for Sterimol calculation.
    4. Identifies the Carbonyl Oxygen as the root atom for consistent 3D conformer generation.
    """
    def __init__(self):
        # Definition of the Michael Addition Product pattern (Ketones, Esters, Aldehydes)
        # Pattern: O=C-[CH]-[CH]-S
        # Mapping indices:
        # 1: O (Carbonyl Oxygen) - used for rooting
        # 2: C (Carbonyl Carbon)
        # 3: C (Alpha Carbon)
        # 4: C (Beta Carbon) - KEY ATOM 1 (Core)
        # 5: S (Thiol Sulfur) - KEY ATOM 2 (Core)
        
        # The SMARTS pattern is flexible to capture various carbonyl environments
        self.product_pattern = Chem.MolFromSmarts('[O:1]=[C:2]-[C:3]-[C:4]-[S:5]')

    def get_core_map(self, mol):
        """
        Locates the core atoms (C_beta, S_thiol) and their substituents.
        """
        if not mol.HasSubstructMatch(self.product_pattern):
            return None
            
        matches = mol.GetSubstructMatches(self.product_pattern)
        match = matches[0] # Take the first match
        
        # Map indices from the match
        idx_O_carbonyl = match[0]
        idx_C_beta = match[3]
        idx_S_thiol = match[4]
        
        # Define core atoms defining the bond and reference point
        core = {
            'c_beta': idx_C_beta,   # Carbon atom (Chalcone side of the bond)
            's_thiol': idx_S_thiol, # Sulfur atom (Thiol side of the bond)
            'root_atom': idx_O_carbonyl # For SMILES canonicalization rooting
        }
        
        # Helper to collect heavy atom neighbors (excluding the bonding partner)
        def get_neighbors(atom_idx, exclude_idx):
            atom = mol.GetAtomWithIdx(atom_idx)
            neighs = []
            for n in atom.GetNeighbors():
                if n.GetIdx() != exclude_idx:
                    neighs.append(n.GetIdx())
            # Sort by mass (heaviest first) to ensure deterministic order of R1, R2...
            neighs.sort(key=lambda x: (-mol.GetAtomWithIdx(x).GetMass()))
            return neighs

        # Build the core map structure
        # 'OnBeta': Substituents attached to Beta Carbon (excluding Sulfur) -> Represents Chalcone remainder
        # 'OnSulfur': Substituents attached to Sulfur (excluding Beta Carbon) -> Represents Thiol remainder
        core_map = {
            'core_atoms': core,
            'subs': {
                'OnBeta': get_neighbors(core['c_beta'], exclude_idx=core['s_thiol']),
                'OnSulfur': get_neighbors(core['s_thiol'], exclude_idx=core['c_beta'])
            },
            'acceptor_type': 'MichaelProduct'
        }
        
        return core_map

# =============================================================================
# STERIMOL DESCRIPTORS L, B1, B5 CALCULATOR
# =============================================================================

class SterimolCalculator:
    """
    Calculates Sterimol parameters (L, B1, B5) along the newly formed C-S bond axis.
    It measures the steric bulk of substituents attached to the Beta Carbon (viewed from Sulfur)
    and substituents attached to the Sulfur (viewed from Beta Carbon).
    """

    def calculate(self, mol, core_map):
        results = {}
        if not core_map:
            return results

        if not MORFEUS_AVAILABLE: # Ensure global flag exists
             return results

        # Retrieve core atom indices
        c_beta_idx = core_map['core_atoms']['c_beta']
        s_thiol_idx = core_map['core_atoms']['s_thiol']

        # Configuration 1: Analyze substituents on C_beta
        # Axis definition: S_thiol -> C_beta. 
        # We measure the steric bulk "behind" the C_beta atom.
        for i, sub_idx in enumerate(core_map['subs']['OnBeta']):
            prefix = f"Prod_Beta_Sub{i+1}"
            self._calc_one(mol, sub_idx, anchor_idx=c_beta_idx, base_idx=s_thiol_idx, 
                           prefix=prefix, results=results)

        # Configuration 2: Analyze substituents on Sulfur
        # Axis definition: C_beta -> S_thiol.
        # We measure the steric bulk "behind" the Sulfur atom.
        for i, sub_idx in enumerate(core_map['subs']['OnSulfur']):
            prefix = f"Prod_Sulfur_Sub{i+1}"
            self._calc_one(mol, sub_idx, anchor_idx=s_thiol_idx, base_idx=c_beta_idx, 
                           prefix=prefix, results=results)

        return results

    def _calc_one(self, mol, sub_idx, anchor_idx, base_idx, prefix, results):
        """
        Calculates Sterimol for a specific substituent using Morfeus.
        """
        try:
            # 1. Identify substituent atoms using BFS
            # CRITICAL: Block both anchor AND base to prevent traversing back across the C-S bond
            sub_atoms = set([sub_idx])
            stack = [sub_idx]
            visited = {sub_idx, anchor_idx, base_idx} 

            while stack:
                curr = stack.pop()
                atom = mol.GetAtomWithIdx(curr)
                for n in atom.GetNeighbors():
                    n_idx = n.GetIdx()
                    if n_idx not in visited:
                        visited.add(n_idx)
                        sub_atoms.add(n_idx)
                        stack.append(n_idx)

            relevant_indices = list(sub_atoms) + [anchor_idx, base_idx]

            # 2. Extract elements and coordinates for the fragment
            elements = []
            coordinates = []
            conf = mol.GetConformer()
            
            # Map original indices to new 1-based indices for Morfeus
            idx_map = {old_idx: i+1 for i, old_idx in enumerate(relevant_indices)}

            for idx in relevant_indices:
                atom = mol.GetAtomWithIdx(idx)
                elements.append(atom.GetSymbol())
                pos = conf.GetAtomPosition(idx)
                coordinates.append([pos.x, pos.y, pos.z])

            new_anchor_idx = idx_map[anchor_idx]
            new_base_idx = idx_map[base_idx]

            # 3. Call Morfeus Sterimol
            # Axis definition: Base -> Anchor (e.g., S -> C_beta)
            # This aligns the measurement axis with the bond direction.
            st = Sterimol(elements, coordinates, new_base_idx, new_anchor_idx)

            results[f"Sterimol_{prefix}_L"] = st.L_value
            results[f"Sterimol_{prefix}_B1"] = st.B_1_value
            results[f"Sterimol_{prefix}_B5"] = st.B_5_value

        except Exception as e:
            # Optionally log error here
            results[f"Sterimol_{prefix}_L"] = np.nan
            results[f"Sterimol_{prefix}_B1"] = np.nan
            results[f"Sterimol_{prefix}_B5"] = np.nan

# =============================================================================
# ELECTRONIC DESCRIPTOR CALCULATOR
# =============================================================================

class ElectronicCalculator:
    """
    Calculates Electronic properties (Gasteiger Charges, Electronegativity) 
    focusing on the C-S bond atoms and their neighbors.
    """

    pauling_electronegativity = {
        1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
        15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    }

    def calculate(self, mol, core_map):
        results = {}
        if not core_map:
            return results

        # Map internal core keys to output descriptor names
        # 'c_beta' -> 'Prod_Beta'
        # 's_thiol' -> 'Prod_Sulfur'
        core_keys_map = {
            'c_beta': 'Prod_Beta',      
            's_thiol': 'Prod_Sulfur'
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

        # Calculate properties for substituents
        for pos_name, subs in core_map['subs'].items():
            # R1 (First substituent)
            if len(subs) > 0:
                self._get_sub_props_extended(mol, subs[0], f"{pos_name}_Sub1", results)

            # R2 (Second substituent)
            if len(subs) > 1:
                self._get_sub_props_extended(mol, subs[1], f"{pos_name}_Sub2", results)

        # Calculate Polarization of the C-S bond
        try:
            if 'Prod_Sulfur' in core_charges and 'Prod_Beta' in core_charges:
                # Difference in charge: Sulfur - Carbon(Beta)
                results['Diff_Charge_S_Cbeta'] = core_charges['Prod_Sulfur'] - core_charges['Prod_Beta']
        except Exception:
            pass

        return results

    def _get_sub_props_extended(self, mol, start_atom_idx, prefix, results_dict):
        """
        Calculates properties for a substituent:
        1. Electronegativity (EN) of the first atom.
        2. Gasteiger Charge of the first atom.
        3. Sum of Gasteiger Charges for the first 3 atoms (local polarity).
        """
        try:
            # BFS to get first 3 atoms of the substituent
            sub_atoms_depth_3 = []
            queue = [(start_atom_idx, 0)] # (idx, depth)
            visited = {start_atom_idx}
            
            idx = 0
            while idx < len(queue) and len(queue) < 3:
                curr, depth = queue[idx]
                idx += 1
                
                atom = mol.GetAtomWithIdx(curr)
                if depth < 2: 
                    for n in atom.GetNeighbors():
                        if n.GetIdx() not in visited:
                            visited.add(n.GetIdx())
                            queue.append((n.GetIdx(), depth + 1))
                            if len(queue) >= 3: break
            
            collected_indices = [x[0] for x in queue]
            
            # First Atom Properties
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
    """ 
    Ensures SMILES are rooted at the Carbonyl Oxygen to ensure consistent 
    graph traversal and folding of the Michael Addition Product. 
    """
    def __init__(self):
        self.finder = ProductCSBondFinder()

    def get_root_atom_idx(self, mol):
        core_map = self.finder.get_core_map(mol)
        if core_map:
            # Returns the Carbonyl Oxygen index identified by the ProductCSBondFinder
            return core_map['core_atoms']['root_atom'] 
        return -1



# Initialize globally
smart_rooter = SmartRooter()

def get_standardized_mol(smiles):
    """Returns (smart_rooted_smiles, mol_object_with_Hs)"""
    try:
        if pd.isna(smiles):
            return None, None

        # Base Molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            return None, None

        # Find rooting point (Carbonyl Oxygen in Product)
        root_idx = smart_rooter.get_root_atom_idx(mol)

        # Generate SMILES
        if root_idx != -1:
            smart_smiles = Chem.MolToSmiles(mol, rootedAtAtom=root_idx, canonical=True, isomericSmiles=True)
        else:
            # UPDATED WARNING MESSAGE HERE
            print(f" ROOTING WARNING: No Product Core (O=C-C-C-S) found for SMILES: {smiles}")
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

        AllChem.MMFFOptimizeMolecule(mol_3d)

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

        canonical_smiles_list = []
        mol_objs = []

        print("Processing molecules (this may take a moment)...")
        for smi in df['SMILES']:
            can, mol = get_standardized_mol(smi)
            canonical_smiles_list.append(can)
            mol_objs.append(mol)

        molecular_descriptors = pd.DataFrame(index=range(len(canonical_smiles_list)))
        # Mordred
        if mordred_features:
            print(" >> Calculating Mordred descriptors...")
            calc = Calculator(descriptors, ignore_3D=False)
            mord_df = calc.pandas(mol_objs)
            mord_df = mord_df.applymap(
                lambda x: np.nan if isinstance(x, (mordred.error.Missing, mordred.error.Error)) else x)
            mord_df = mord_df[sorted(mord_df.columns)]
            molecular_descriptors = pd.concat([molecular_descriptors, mord_df], axis=1)
            print(f" OK: Mordred: {mord_df.shape}")

        # Morgan fingerprints
        if fingerprints:
            print(" >> Calculating Morgan fingerprints...")
            fingerprints_counts = []
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
            for mol in mol_objs:
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
            
            print(" >> Calculating Product C-S Bond Sterimol & Electronic descriptors...")

            finder = ProductCSBondFinder()
            
            sterimol_calc = SterimolCalculator()
            electronic_calc = ElectronicCalculator()

            descriptor_data = []
            for mol in mol_objs:
                desc = {}
                if mol:
                    core_map = finder.get_core_map(mol)
                    if core_map:
                        st_res = sterimol_calc.calculate(mol, core_map)
                        desc.update(st_res)
                        el_res = electronic_calc.calculate(mol, core_map)
                        desc.update(el_res)

                descriptor_data.append(desc)

            try:
                st_df = pd.DataFrame(descriptor_data)
                
                # =========================================================
                # NEW: CALCULATE INTERACTIONS (CROSS-TERMS)
                # =========================================================
                print(" >> Generating Interaction Terms (Beta x Sulfur)...")
                
                # 1. STERIMOL INTERACTIONS (Geometry)
                # Find columns for Beta & Sulfur (L, B1, B5)
                beta_cols = [c for c in st_df.columns if 'Prod_Beta' in c and 'Sterimol' in c]
                sulfur_cols = [c for c in st_df.columns if 'Prod_Sulfur' in c and 'Sterimol' in c]
                
                interaction_count = 0
                for b_col in beta_cols:
                    for s_col in sulfur_cols:

                        if any(x in b_col for x in ['_L', '_B1', '_B5']) and any(x in s_col for x in ['_L', '_B1', '_B5']):
                            new_col_name = f"INTER_{b_col}_x_{s_col}"
                            st_df[new_col_name] = st_df[b_col] * st_df[s_col]
                            interaction_count += 1

                # 2. ELECTRONIC INTERACTIONS (Electrostatics)
                # Eg. Prod_Beta_Charge * Prod_Sulfur_Charge (charge interaction)
                # Eg. Prod_Beta_Sub1_EN * Prod_Sulfur_Sub1_EN (differences in electronegativity)
                el_beta_cols = [c for c in st_df.columns if 'Prod_Beta' in c and ('Charge' in c or 'EN' in c)]
                el_sulfur_cols = [c for c in st_df.columns if 'Prod_Sulfur' in c and ('Charge' in c or 'EN' in c)]
                
                for b_col in el_beta_cols:
                    for s_col in el_sulfur_cols:
                         new_col_name = f"EL_INTER_{b_col}_x_{s_col}"
                         st_df[new_col_name] = st_df[b_col] * st_df[s_col]
                         interaction_count += 1

                print(f"    -> Generated {interaction_count} interaction features.")
                # =========================================================

                # Merge into main DF
                st_df.index = molecular_descriptors.index
                molecular_descriptors = pd.concat([molecular_descriptors, st_df], axis=1)
                print(f" OK: Added {st_df.shape[1]} Product C-S features (including interactions)")
            except Exception as e:
                print(f" Sterimol + Electronic error: {e}")

        # ChemBERTa
        if chembert:
            print(">> Calculating ChemBERTa embeddings...")
            embedder = ChemBERTaEmbedder()
            if embedder.available:
                embs = [embedder.get_embedding(smi) for smi in canonical_smiles_list]
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
                canonical_smiles_list, circus_params=circus_params, chyline_params=chyline_params
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

    return molecular_descriptors_cleaned.reset_index(drop=True), canonical_smiles_list


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