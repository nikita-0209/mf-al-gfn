import glob
import os
import re
import shutil
import subprocess
import time
import traceback
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# TODO: fix shutil.Error: Destination path already exists




def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def get_files_in_dir(dir, specs=None):
    if specs is None:
        return natural_sort(glob.glob(os.path.join(dir, "*")))
    else:
        return natural_sort(glob.glob(os.path.join(dir, specs)))


def get_rdkit_ff_coordinates(
    mol,
    FF="MMFF",
    filename=None,
    filepath=None,
    save_min_energy=False,
    xyzblock=False,
    return_mol=False,
    conformer_config=None,
    **kwargs
):
    """
    Takes an input molecule and convert to optimized 3D coordinates, can also save coordinates and call to write Orca/Gaussian input files

    :param mol: molecule in Mol format, or SMILES string
    :param FF: Molecular mechanics forcefiled
    :param num_conf: number of configurations to generate, 50 for num_rot_bonds < 7; 200 for 8 <= num_rot_bonds <= 12; 300 otherwise
    :param filename: saves .xyz file with this filename
    :param filepath: ^ but filepath
    :param maxattempts: max attempts at embedding conformer
    :param randomcoords: whether to use random coordinates
    :param prunermsthres: whether to use a RMSD threshold to keep some conformers only
    :param xyzblock: returns xyzblock
    :param return_mol: returns molecule
    """
    start_time = time.time()
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    Chem.SanitizeMol(mol)
    mol_h = Chem.AddHs(mol)

    if conformer_config is None:
        conformer_config = {
            "num_conf": 4,
            "maxattempts": 100,
            "randomcoords": True,
            "prunermsthres": 1,
        }
    num_conf = conformer_config["num_conf"]
    prunermsthres = conformer_config["prunermsthres"]
    randomcoords = conformer_config["randomcoords"]
    maxattempts = conformer_config["maxattempts"]

    AllChem.EmbedMultipleConfs(
        mol_h,
        numConfs=num_conf,
        pruneRmsThresh=prunermsthres,
        maxAttempts=maxattempts,
        useRandomCoords=randomcoords,
    )  # prunermsthres appear to cause issues later
    num_conf = mol_h.GetNumConformers()  # get new number after pruning
    conformer_generation_time = time.time() - start_time
    if FF.lower() == "mmff":
        try:
            msg = [
                AllChem.MMFFOptimizeMolecule(mol_h, confId=i, maxIters=1000)
                for i in range(num_conf)
            ]
            # print(msg, Chem.MolToSmiles(mol))
        except Exception as e:
            print(e)
        mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
        mi = np.argmin(
            [
                AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy()
                for i in range(num_conf)
            ]
        )
    elif FF.lower() == "uff":
        try:
            msg = [
                AllChem.UFFOptimizeMolecule(mol_h, confId=i, maxIters=1000)
                for i in range(num_conf)
            ]
            # print(msg, Chem.MolToSmiles(mol))
        except Exception as e:
            print(e)
        mi = np.argmin(
            [
                AllChem.UFFGetMoleculeForceField(mol_h, confId=i).CalcEnergy()
                for i in range(num_conf)
            ]
        )
    else:
        raise NotImplementedError
    xyz_file = None
    if filepath is not None and filename is not None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        xyz_file = os.path.join(filepath, str(filename) + "_" + FF + "_opt.xyz")
        Chem.MolToXYZFile(mol_h, xyz_file, confId=int(mi))  # save xyz
        if save_min_energy:
            min_energy = np.min(
                [
                    AllChem.UFFGetMoleculeForceField(mol_h, confId=i).CalcEnergy()
                    for i in range(num_conf)
                ]
            )
            print("minimum energy is " + str(min_energy))
            with open(
                os.path.join(filepath, str(filename) + "_" + FF + "_opt_energy.txt"),
                "w",
            ) as f:
                f.write("total energy \n" + str(min_energy))
                f.write(
                    "\n conformer generation time \n" + str(conformer_generation_time)
                )
                f.write("\n total optimization time \n")
                f.write(str(time.time() - start_time))
                f.close()
    if xyzblock:
        return Chem.MolToXYZBlock(mol_h, confId=int(mi))  # return xyz
    if return_mol:
        return mol_h, int(mi)  # .GetConformer(id=int(mi)).GetOwningMol() # return mol
    print("total FF optimization time is", time.time() - start_time)
    return xyz_file  # return xyz_file name


def run_gfn_xtb(
    filepath,
    filename=None,
    gfn_version="2",
    opt=True,
    gfn_xtb_config: str = None,
    coord_file_format="xyz",
    optimized_xyz_dir=None,
    log_file_dir="log_gfn_xtb",
    **kwargs
):
    """
    Runs GFN_FF given a directory and either a coord file or all coord files will be run

    :param filepath: Directory containing the coord file
    :param filename: if given, the specific coord file to run
    :param gfn_version: GFN_xtb version (default is 2)
    :param opt: optimization or singlet point (default is opt)
    :param gfn_xtb_config: additional xtb config (default is None)
    :param coord_file_format: coordinate file format if all coord files in filepath is run (default is xyz)
    :return:
    """
    if filename is None:
        xyz_files = get_files_in_dir(filepath, "*." + coord_file_format)
    else:
        xyz_files = [os.path.join(filepath, filename)]

    if opt:
        opt = "--opt"
    else:
        opt = ""
    starting_dir = os.getcwd()
    os.chdir(filepath)
    log_file_dir = os.path.join(filepath, log_file_dir)
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)
    for xyz_file in xyz_files:
        file_name = os.path.splitext(xyz_file)[0]
        cmd = "xtb --gfn {} {} {} {}".format(
            str(gfn_version), xyz_file, opt, str(gfn_xtb_config or "")
        )
        with open(file_name + ".out", "w") as fd:
            subprocess.run(cmd, shell=True, stdout=fd, stderr=subprocess.STDOUT)
        # cmd = "xtb --gfn {} {} {} {} 2>&1 | tee -a {}".format(str(gfn_version), xyz_file, opt, str(gfn_xtb_config or ''), file_name+'.out')  # xtb is weird in output redirection
        # subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
        if os.path.isfile(os.path.join(filepath, "NOT_CONVERGED")):
            # todo alternatively try gfn0-xtb and then gfn2-xtb
            warnings.warn(
                "xtb --gfn {} for {} is not converged, using last optimized step instead; proceed with caution".format(
                    str(gfn_version), file_name
                )
            )
            shutil.move(
                os.path.join(filepath, "xtblast.xyz"),
                os.path.join(
                    log_file_dir, os.path.basename(file_name) + "_xtb_opt.xyz"
                ),
            )
            shutil.move(
                os.path.join(filepath, "NOT_CONVERGED"),
                os.path.join(log_file_dir, "NOT_CONVERGED"),
            )
        elif not os.path.isfile(
            os.path.join(filepath, "xtbopt.xyz")
        ):  # other abnormal convergence:
            warnings.warn(
                "xtb --gfn {} for {} abnormal termination, likely scf issues, using initial geometry instead; proceed with caution".format(
                    str(gfn_version), file_name
                )
            )
            shutil.copy2(
                xyz_file,
                os.path.join(
                    log_file_dir, os.path.basename(file_name) + "_xtb_opt.xyz"
                ),
            )
        else:
            shutil.move(
                os.path.join(filepath, "xtbopt.xyz"),
                os.path.join(
                    log_file_dir, os.path.basename(file_name) + "_xtb_opt.xyz"
                ),
            )
            try:
                shutil.move(
                    os.path.join(filepath, "xtbopt.log"),
                    os.path.join(log_file_dir, "xtbopt.log"),
                )
                shutil.move(
                    os.path.join(filepath, "xtbtopo.mol"),
                    os.path.join(log_file_dir, "xtbtopo.mol"),
                )
                shutil.move(
                    os.path.join(filepath, "wbo"), os.path.join(log_file_dir, "wbo")
                )
                shutil.move(
                    os.path.join(filepath, "charges"),
                    os.path.join(log_file_dir, "charges"),
                )
                shutil.move(
                    os.path.join(filepath, "xtbrestart"),
                    os.path.join(log_file_dir, "restart"),
                )
            except Exception:
                # some versions of xtb do not produce these files
                traceback.print_exc()
                pass
            print("{} xtb optimization is done".format(file_name))
        shutil.move(file_name + ".out", log_file_dir)
        if optimized_xyz_dir:
            if not os.path.exists(optimized_xyz_dir):
                os.makedirs(optimized_xyz_dir)  # make directory to save xyz file
            shutil.copy2(
                os.path.join(
                    log_file_dir, os.path.basename(file_name) + "_xtb_opt.xyz"
                ),
                os.path.join(
                    optimized_xyz_dir, os.path.basename(file_name) + "_xtb_opt.xyz"
                ),
            )
    os.chdir(starting_dir)
