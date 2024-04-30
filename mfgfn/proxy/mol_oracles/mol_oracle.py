import os

import torch
from gflownet.proxy.base import Proxy

from mfgfn.proxy.mol_oracles.ipea_xtb import XTB_IPEA

"""
To run, install xtb via conda, or via source (+ export PATH="~/xtb/bin/:$PATH")

A given, closed-shell organic molecule has paired electrons. 
Energy is required to kick out an electron, energy will be gained when you add an electron
The lowest energy required to kick out an electron is the ionization potential (IP)
The maximum energy gained upon receiving an electron is the electron affinity (EA)

These are assuming there are no changes in geometry upon receiving/adding an electron (i.e. vertical energy) 
If you allow the molecule to rearrange upon losing/receiving an electron, then that is adiabatic IP/EA

By optimizing IP/EA, you can tailor a molecule to be a suitable (photo)catalyst/semiconductor/electrolyte. 
We here use two simple tasks to show by proof-of-concept that you can design the electronics of conjugated molecules

There are multiple different oracles implemented here: 
Oracle 1: RDKIT MMFF geometry opt + XTB IPEA single point (neutral, vertical)
Oracle 2: RDKIT MMFF geometry opt + XTB geometry optimization (neutral) + XTB IPEA single point (vertical) 
Oracle 3: RDKIT MMFF geometry opt + XTB geometry optimization (neutral) + XTB geometry optimization (ionic)

There are 2 different tasks:
# Task 1: minimize adiabatic IP
# Task 2: minimize adiabatic EA

If we want to involve DFT, (they are very slow), then we can slightly modify for a task on Eg or VIP/EA 

We are taking multiple short-cuts here, e.g. not searching for conformers properly. To be written.

To distribute the number of threads reasonable in the OpenMP section for XTB it is recommended to use 
export OMP_NUM_THREADS=<ncores>,1
export OMP_STACKSIZE=4G

Concern: maybe IPEA don't depend that much geometry and this only adds noise (and possibly limited noise)
Possible solution if that's the case: sample more geometry via conformer_ladder; or use DFT (which has bias and noise)
"""

default_config = {
    "task": "ip",  # or ea
    "oracle_config": {
        "log_dir": os.getcwd(),
        "moltocoord_config": {
            "conformer_config": {
                "num_conf": 2,
                "maxattempts": 100,
                "randomcoords": True,
                "prunermsthres": 1.5,
            },
        },
        "conformer_ladder": 0,  # on each oracle ladder, use ladder^x more conformers (0 -> no changes)
        "remove_scratch": True,
        "ff": "mmff",
        "semiempirical": "xtb",
        "mol_repr": "selfies",  # or smiles
    },
}


class MoleculeOracle(Proxy):
    def __init__(self, cost, task, oracle_config, oracle_level=None, **kwargs):
        super().__init__(**kwargs)

        self.cost = cost
        self.task = task
        self.oracle_level = (
            oracle_level  # without change, this requires highest fidelity
        )
        oracle_config.log_dir = os.getcwd()
        self.xtb_ipea = XTB_IPEA(task=self.task, **oracle_config)

    def __call__(self, mols, *args, **kwargs):
        scores = [self.xtb_ipea(mol, self.oracle_level) for mol in mols]
        scores = torch.tensor(scores, device=self.device, dtype=self.float)
        return scores

    def replaceNaN(self, mols, scores):
        # consider only those mols which have score which is not nan
        mols = [mols[i] for i in range(len(mols)) if not torch.isnan(scores[i])]
        return mols, scores[~torch.isnan(scores)]
