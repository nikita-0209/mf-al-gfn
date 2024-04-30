import os
import random
import shutil
import string
import traceback

import selfies as sf
from rdkit import Chem

from mfgfn.proxy.mol_oracles.geom_opt import get_rdkit_ff_coordinates, run_gfn_xtb


class MolToCoord:
    def __init__(self, config):
        self.log_dir = config.get("log_dir", os.getcwd())
        os.makedirs(self.log_dir, exist_ok=True)
        self.conformer_config = config.get("conformer_config", None)
        self.semiempirical_config = config.get(
            "semiempirical_config", {"log_file_dir": "log_gfn_xtb"}
        )

    def _rdkit_ff_wrapper(self, mol, save_dir, mol_name, level):
        print("converting to {} coordinates for {}".format(level, mol_name))
        coord_file = get_rdkit_ff_coordinates(
            mol,
            FF=level,
            conformer_config=self.conformer_config,
            filepath=save_dir,
            filename=mol_name,
        )
        return coord_file

    def _gfn_xtb_wrapper(self, save_dir, input_coord, charge=None):
        print("converting to gfn_xtb coordinates for {}".format(input_coord))
        if charge is not None:
            charge = "--chrg {}".format(str(charge))
        run_gfn_xtb(
            filepath=save_dir,
            filename=input_coord,
            gfn_xtb_config=charge,
            optimized_xyz_dir=save_dir,
            **self.semiempirical_config
        )

    def optimize_molecule(
        self,
        mol,
        level,
        charge=None,
        mol_name=None,
        mol_repr="selfies",
        *args,
        **kwargs
    ):
        mol_name = mol_name or "".join(
            random.choices(string.ascii_uppercase + string.digits, k=15)
        )
        os.chdir(self.log_dir)
        save_dir = os.path.join(self.log_dir, mol_name)
        # print('save_dir is', save_dir)
        os.makedirs(save_dir, exist_ok=True)
        final_coord = mol_name + "_" + str(level) + "_opt.xyz"

        if os.path.isfile(mol):
            # do optimization based on given mol file
            pass
        elif isinstance(mol, Chem.Mol) or mol_repr == "mols":
            # do optimization based on Chem.Mol object
            pass
        elif mol_repr == "smiles":
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return None, None, None, None
        elif mol_repr == "selfies" or len(smiles) < 10:
            smiles = sf.decoder(mol)
            if smiles is None:
                return None, None, None, None
            mol = Chem.MolFromSmiles(smiles)

        # get geometry by gfn-ff, mmff/uff, or rdkit
        try:
            os.chdir(save_dir)
            if level == "mmff" or level == "uff":
                ff_coord_file = self._rdkit_ff_wrapper(mol, save_dir, mol_name, level)
                try:
                    shutil.copy2(ff_coord_file, os.path.join(save_dir, final_coord))
                    # shutil.copy2(os.path.join(save_dir, ff_coord_file),
                    #  os.path.join(save_dir, final_coord))
                except shutil.SameFileError:
                    pass
                log_file = None
            elif level == "xtb":
                # todo mol must be a file; else do mmff/uff based on string first (removed in this simplified version)
                try:
                    shutil.copy2(mol, save_dir)
                except shutil.SameFileError:
                    pass
                mol = os.path.basename(mol)
                self._gfn_xtb_wrapper(save_dir, mol, charge)
                try:
                    shutil.copy2(
                        os.path.join(
                            save_dir, os.path.splitext(mol)[0] + "_xtb_opt.xyz"
                        ),
                        os.path.join(save_dir, final_coord),
                    )
                except shutil.SameFileError:
                    pass
                print("semiempirical geometry conversion for {} is done".format(mol))
                log_file = os.path.join(
                    save_dir,
                    self.semiempirical_config["log_file_dir"],
                    os.path.splitext(mol)[0] + ".out",
                )
            else:
                raise NotImplementedError

            os.chdir(self.log_dir)
            return save_dir, final_coord, log_file, mol_name
        except Exception:
            # other parts already deal with: if one ff_opt fails; if semi-empirical opt fails; retry with conf embedding
            # occasionally, however, all embedding fails, or every ff geometry fails, and we need to catch this
            traceback.print_exc()
            return None, None, None, None

    def __call__(
        self,
        mol,
        level,
        charge=None,
        mol_name=None,
        mol_repr="selfies",
        *args,
        **kwargs
    ):
        return self.optimize_molecule(
            mol=mol, level=level, charge=charge, mol_name=mol_name, mol_repr=mol_repr
        )
