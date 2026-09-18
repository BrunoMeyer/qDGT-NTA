"""Feature definitions used when the bundled XGBoost models were trained."""
from functools import lru_cache
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.warning")
CALCULATOR = Calculator(descriptors, ignore_3D=True)


def _bits(fp):
    result = np.zeros(fp.GetNumBits(), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, result)
    return result


def _rdkit_descriptors(mol):
    values = np.asarray([function(mol) for _, function in Descriptors._descList], dtype=float)
    values[~np.isfinite(values)] = np.nan
    values = np.clip(values, -1e10, 1e10)
    fill = np.nanmean(values) if not np.isnan(values).all() else 0.0
    values = np.nan_to_num(values, nan=fill)
    # The supplied models fit a scaler separately to each molecule's descriptor vector.
    return StandardScaler().fit_transform(values.reshape(-1, 1)).ravel()


def _mordred_descriptors(mol):
    return np.asarray(list(CALCULATOR(mol).fill_missing().values()), dtype=float)


@lru_cache(maxsize=4096)
def generate_features(smiles, model):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES is empty")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if model == "d":
        return np.concatenate((
            _rdkit_descriptors(mol),
            _bits(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)),
            _bits(MACCSkeys.GenMACCSKeys(mol)),
            _count_fingerprint(mol),
        ))
    if model == "pos":
        return np.concatenate((_rdkit_descriptors(mol), _mordred_descriptors(mol)))
    if model == "neg":
        return np.concatenate((_count_fingerprint(mol), _mordred_descriptors(mol)))
    raise ValueError(f"Unknown model: {model}")


def _count_fingerprint(mol):
    fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=3, nBits=1024)
    result = np.zeros(1024, dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, result)
    return result
