import re
import os
import pickle
import pandas as pd
from typing import Iterable, Any


def shingles(value: str) -> Iterable[str]:
    """
    Generate multi-word tokens delimited by punctuation.
        Parameters
        ----------
        value : str
            The value to shingle.

        Returns
        -------
        Iterable[str]
            A generator of shingles.
    """

    delimiterPattern = re.compile(r"[^\w\s\-_@&]+")
    for shingle in delimiterPattern.split(value):
        yield re.sub(r"\s+", " ", shingle.strip().lower())


def is_numeric(values: Iterable[Any]) -> bool:
    """
    Check if a given column contains only numeric values.

    Parameters
    ----------
    values :  Iterable[Any]
        A collection of values.

    Returns
    -------
    bool
        All non-null values are numeric or not (True/False).

    """
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    return pd.api.types.is_numeric_dtype(values.dropna())


def pickle_python_object(obj: Any, object_path: str):
    """
    Save the given Python object to the given path.

    Parameters
    ----------
    obj : Any
        Any *picklable* Python object.
    object_path : str
        The path where the object will be saved.

    Returns
    -------

    """
    parent_dir = "/".join(object_path.split("/")[:-1])
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    try:
        with open(object_path, "wb") as save_file:
            pickle.dump(obj, save_file)
    except Exception:
        raise pickle.PicklingError(
            "Failed to save object {} to {}!".format(str(obj), object_path)
        )


def unpickle_python_object(object_path: str) -> Any:
    """
    Load the Python object from the given path.

    Parameters
    ----------
    object_path : str
        The path where the object is saved.

    Returns
    -------
    Any
        The object existing at the provided location.

    """
    if not os.path.isfile(object_path):
        raise FileNotFoundError("File {} does not exist locally!".format(object_path))
    try:
        with open(object_path, "rb") as save_file:
            obj = pickle.load(save_file)
    except Exception:
        raise pickle.UnpicklingError(
            "Failed to load object from {}!".format(object_path)
        )
    return obj
