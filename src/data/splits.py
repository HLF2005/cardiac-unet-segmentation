"""Reproducible patient-level dataset splitting."""

from pathlib import Path

import numpy as np


def create_patient_split(
    data_dir,
    seed=42,
    train_patients=70,
    valid_patients=15,
):
    patient_dirs = sorted(
        path for path in Path(data_dir).iterdir()
        if path.is_dir()
    )

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(patient_dirs)

    train_end = train_patients
    valid_end = train_patients + valid_patients

    return (
        patient_dirs[:train_end],
        patient_dirs[train_end:valid_end],
        patient_dirs[valid_end:],
    )
