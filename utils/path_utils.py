'''
This module will define some path utils
such as building paths to load data, ...

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 02/02/2024
'''

import os
from pathlib import Path


def get_run_path(
        root_dir: str,
        subject_nr: int,
        session_nr: int,
        run_nr: int,
        empty_dir=False) -> Path:
    '''
    This method will build the path
    for a @hoffmann_efficient_2008's run.

    Example:
    ```
    root_dir = "data/OriginalDataEPFL/"
    get_run_path(root_dir, 1, 1, 2)
    ----- Output -----
    PosixPath('data/OriginalDataEPFL/subject1/session1/eeg_200605191430_epochs.mat')
    ```

    :param root_dir: Path to the directory
    with all the @hoffmann_efficient_2008 subjects.
    :param subject_nr: Subject number to load.
    :param session_nr: Session number to load.
    :param run_nr: Run number to load.
    :param empty_dir: Whether `root_dir` is empty or
    not.
    :return Path: Full run's path in a `pathlib.Path`
    format.
    '''
    path = Path(root_dir) / f"subject{subject_nr}" /\
        f"session{session_nr}"

    if empty_dir:
        run_name = f"run_{run_nr - 1}.pkl"
    else:
        run_name = sorted(os.listdir(path))[run_nr - 1]

    return path / run_name
