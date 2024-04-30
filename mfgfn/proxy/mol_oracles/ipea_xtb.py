import math
import os
import re
import shutil
import subprocess
import traceback

import numpy as np

from mfgfn.proxy.mol_oracles.mol_to_coord import MolToCoord


def hartree2ev(hartree):
    return hartree * 27.2114


class XTB_IPEA:
    def __init__(
        self,
        task,
        log_dir,
        moltocoord_config,
        ff="mmff",
        semiempirical="xtb",
        semiempirical_version=2,
        mol_repr="selfies",
        remove_scratch=True,
        conformer_ladder=0,
    ):
        self.task = task
        self.charged_species = "+1" if task == "ip" else "-1"
        self.log_dir = log_dir
        self.ff = ff
        self.semiempirical = semiempirical
        self.semiempirical_version = semiempirical_version
        self.conformer_ladder = conformer_ladder
        if self.semiempirical == "xtb":
            if self.semiempirical_version == 2:
                self.correction_factor = 4.8455  # for GFN-XTB2, which is the default
            else:
                raise NotImplementedError

        self.mol_repr = mol_repr

        self.moltocoord_config = moltocoord_config
        self.moltocoord_config["log_dir"] = self.log_dir
        self.mol_to_coord = MolToCoord(self.moltocoord_config)
        self.inital_num_conf = self.moltocoord_config.get("conformer_config", {}).get(
            "num_conf", 2
        )
        self.remove_scratch = remove_scratch

    def __call__(self, molecule, oracle_level=None, *args, **kwargs):
        return self.get_score(molecule, oracle_level)

    def get_score(self, molecule, oracle_level=None):
        os.chdir(self.log_dir)
        # run to get mmff geometry from string; will also check is molecule is valid (else return None)
        num_conf = math.floor(
            self.inital_num_conf * (oracle_level**self.conformer_ladder)
        )
        self.mol_to_coord.conformer_config["num_conf"] = num_conf

        save_dir, mol_coord, _, mol_name = self.mol_to_coord(
            molecule, level=self.ff, mol_repr=self.mol_repr
        )
        if save_dir is None or mol_coord is None:
            return np.nan
        try:
            if oracle_level == 1:
                # we use mmff geometry, get the vertical IP/EA
                score = self._get_vipea(save_dir, mol_coord)
            elif oracle_level == 2:
                # we use mmff to optimize to XTB geometry, then get the vertical IP/EA
                save_dir, mol_coord, _, _ = self.mol_to_coord(
                    os.path.join(save_dir, mol_coord),
                    level=self.semiempirical,
                    mol_name=mol_name,
                )
                score = self._get_vipea(save_dir, mol_coord)
            else:  # oracle_level == 3:
                # we use mmff to optimize to XTB geometry at neutral geometry, then at charged geometry
                save_dir, mol_coord, neutral_log, _ = self.mol_to_coord(
                    os.path.join(save_dir, mol_coord),
                    level=self.semiempirical,
                    mol_name=mol_name,
                )
                mol_name = mol_name + "_ionic"
                save_dir, mol_coord, ionic_log, _ = self.mol_to_coord(
                    os.path.join(save_dir, mol_coord),
                    charge=self.charged_species,
                    level=self.semiempirical,
                    mol_name=mol_name,
                )
                # then we process the difference to get adiabatic IP/EA
                score = self._get_aipea(save_dir, [neutral_log, ionic_log])
        except Exception:
            traceback.print_exc()
            os.chdir(self.log_dir)
            return np.nan

        os.chdir(self.log_dir)
        if self.remove_scratch:
            shutil.rmtree(save_dir)
            if os.path.isdir(save_dir.replace("_ionic", "")):
                shutil.rmtree(save_dir.replace("_ionic", ""))
        return score

    def _get_vipea(self, save_dir, mol_coord):
        # vertical ionization potential and electron affinity
        ipea_save_dir = os.path.join(save_dir, "log_ipea_xtb")
        if not os.path.exists(ipea_save_dir):
            os.makedirs(ipea_save_dir)
        shutil.copy2(os.path.join(save_dir, mol_coord), ipea_save_dir)
        os.chdir(ipea_save_dir)
        score = self._run_vipea_calc(mol_coord)
        return score

    def _run_vipea_calc(self, coord_file):
        cmd = "xtb {} --gfn {} --v{}".format(
            coord_file, str(self.semiempirical_version), self.task
        )  # self.task=ip/ea/ipea
        with open("ipea_xtb.out", "w") as fd:
            subprocess.run(cmd, shell=True, stdout=fd, stderr=subprocess.STDOUT)
        with open("ipea_xtb.out") as f:
            for l in f:
                if "delta SCC IP (eV):" in l:
                    score = float(re.search(r"[+-]?(?:\d*\.)?\d+", l).group())
                elif "delta SCC EA (eV):" in l:
                    score = float(re.search(r"[+-]?(?:\d*\.)?\d+", l).group())
        return score

    def _get_aipea(self, save_dir, coord_files):
        neutral_log = coord_files[0]
        charged_log = coord_files[1]
        # neutral_log = os.path.join(save_dir, coord_files[0])
        # charged_log = os.path.join(save_dir, coord_files[1])

        def _get_energy(file):
            with open(file) as f:
                for l in f:
                    if "TOTAL ENERGY" in l:
                        energy = float(re.search(r"[+-]?(?:\d*\.)?\d+", l).group())
            return energy

        if self.task == "ip":
            return (
                hartree2ev(_get_energy(charged_log) - _get_energy(neutral_log))
                - self.correction_factor
            )
        else:
            return (
                hartree2ev(_get_energy(neutral_log) - _get_energy(charged_log))
                - self.correction_factor
            )
