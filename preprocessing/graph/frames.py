#
from preprocessing.protein_chemistry import atom_type_mass , dictionary_covalent_bonds_numba, list_atoms, aa_to_index
from numba import njit, types
from numba.typed import List, Dict
import numpy as np


def get_atom_frameCloud(sequence, atom_coordinates, atom_ids):
    atom_clouds = np.concatenate(atom_coordinates, axis=0)
    atom_attributes = np.concatenate(atom_ids, axis=-1)
    atom_triplets = np.array(_get_atom_triplets(sequence, atom_ids, dictionary_covalent_bonds_numba),
                             dtype=np.int32)
    atom_indices = np.concatenate([np.ones(len(atom_ids[l]), dtype=np.int32) * l for l in range(len(sequence))],
                                  axis=-1)[:, np.newaxis]
    return atom_clouds, atom_triplets, atom_attributes, atom_indices


def _get_atom_triplets(sequence, atom_ids, dictionary_covalent_bonds_numba):
    r"""
    create
      { prev-residue l-1 }  {  curr-residue l }  { next-residue l+1 }
                        C <= N* => CA   <=  C* =>  N
    parameters:
      sequence:
      atom_ids:
    output:
      atom_triplets:
    """
    L = len(sequence)
    atom_triplets = [] # List()
    all_keys = list(dictionary_covalent_bonds_numba.keys()) # List(dictionary_covalent_bonds_numba.keys() )
    current_natoms = 0
    for l in range(L): # per residue
      aa = sequence[l]
      atom_id = atom_ids[l]
      natoms = len(atom_id)
      for n in range(natoms): # per atom of residue
        id = atom_id[n]
        # atom N, special case, bound to C of previous aa.
        if (id == 17):
          # N's previous is C of previous residue, or -1 if there's no previous residue
          if l > 0:
            if 0 in atom_ids[l - 1]:
              previous = current_natoms - len(atom_ids[l - 1]) + atom_ids[l - 1].index(0)
            else:
              previous = -1
          else:
            previous = -1
          # N's next is Ca of current residue
          if 1 in atom_id:
            next = current_natoms + atom_id.index(1)
          else:
            next = -1
        # atom C, special case, bound to N of next aa.
        elif (id == 0):
          # C's previous is N of current residue
          if 1 in atom_id:
            previous = current_natoms + atom_id.index(1)
          else:
            previous = -1
          # C's next is N of next residue, or -1 if there's no next residue
          if l < L - 1:
            if 17 in atom_ids[l + 1]:
              next = current_natoms + natoms + atom_ids[l + 1].index(17)
            else:
              next = -1
          else:
            next = -1
        else:
          key = (aa + '_' + str(id) )
          if key in all_keys:
            previous_id, next_id, _ = dictionary_covalent_bonds_numba[(aa + '_' + str(id) )] 
          else:
            print('Strange atom', (aa + '_' + str(id) ))
            previous_id = -1
            next_id = -1
          if previous_id in atom_id: # if prev atom id(covalent bond) in current residue
              previous = current_natoms + atom_id.index(previous_id) # in which position of residue (not atom index)
          else:
              previous = -1
          if next_id in atom_id: # if prev atom id(covalent bond) in current residue
              next = current_natoms + atom_id.index(next_id)
          else:
              next = -1
        atom_triplets.append((current_natoms + n, previous, next))
      current_natoms += natoms
    return atom_triplets


def get_aa_frameCloud(atom_coordinates, atom_ids, verbose=True, method='triplet_backbone'):
  r"""
  create the coordinate of residues and triplets
  
  parameters:
    atom_coordinates: List(np.array), list shape = (num_residues,), np.array.shape = (num_heavy_atoms_in_residue, 3)
    atom_ids: List(List), outer list shape = (num_residues,), inner list shape = (num_heavy_atoms_in_residue,)
  outputs:
    aa_indices:   (num_residues, 1), value = [[0], [1], [2], ...., [num_residues-1]]
    aa_clouds:    (1 + 2 * num_residues, 3)
    aa_triplets:  (num_residues, 3)
  """
  aa_clouds, aa_triplets = _get_aa_frameCloud_triplet_sidechain(atom_coordinates, atom_ids, verbose=verbose)
  aa_indices  = np.arange(len(atom_coordinates)).astype(np.int32)[:, np.newaxis] # np.array, shape = (num_residues, 1)
  aa_clouds   = np.array(aa_clouds) # (1 + 2 * num_residues, 3)
  aa_triplets = np.array(aa_triplets, dtype=np.int32) # (num_residues, 3)
  return aa_clouds, aa_triplets, aa_indices

# 
# @njit(cache=True, parallel=False)
def _get_aa_frameCloud_triplet_sidechain(atom_coordinates, atom_ids, verbose=True):
  r"""
  create the coordinate of residues and triplets
  
  parameters:
    atom_coordinates: List(np.array), list shape = (num_residues,), np.array.shape = (num_heavy_atoms_in_residue, 3)
    atom_ids: List(List), outer list shape = (num_residues,), inner list shape = (num_heavy_atoms_in_residue,)
              atom id is atom index of "preprocessing.protein_chemistry.list_atoms"
  outputs:
    aa_clouds: List(np.array), shape = (1 + 2 * num_residues, 3)
               store all coord, ex, [1st_res_ctr_coord, 1st_res_prev_coord, 1st_res_next_coord, ...]
    aa_triplets: List(tuple), list shape = (num_residues,)
                 record "corresponding center/prev/next coord's index in aa_clouds", ex, (0,1,2) where 0 indicate index-0 of aa_clouds
  """
  # get residues number of a protein
  L = len(atom_coordinates)
  aa_clouds = []
  aa_triplets = []
  count = 0
  for l in range(L):
    # for each residue
    atom_coordinate = atom_coordinates[l] # np.array, shape = (num_heavy_atoms_in_residue, 3)
    atom_id = atom_ids[l] # list, shape = (num_heavy_atoms_in_residue,)
    natoms = len(atom_id)
    # get coord of C_alpha(CA) in this residue, CA's index is 1
    if 1 in atom_id:
      calpha_coordinate = atom_coordinate[atom_id.index(1)]
    else:
      if verbose:
        print('Warning, pathological amino acid missing calpha', l)
      calpha_coordinate = atom_coordinate[0]

    center = 1 * count
    aa_clouds.append(calpha_coordinate)
    count += 1
    if count > 1:
      previous = aa_triplets[-1][0]
    else: # for first residue
      # Need to place another virtual Calpha.
      virtual_calpha_coordinate = 2 * calpha_coordinate - atom_coordinates[1][0]
      aa_clouds.append(virtual_calpha_coordinate)
      previous = 1 * count
      count += 1

    sidechain_CoM = np.zeros(3, dtype=np.float32)
    sidechain_mass = 0.
    # when computing sidechain_CoM, we exclude atoms such as C CA N O OXT
    for n in range(natoms):
      if not atom_id[n] in [0, 1, 17, 26, 34]:
        mass = atom_type_mass[atom_id[n]]
        sidechain_CoM += mass * atom_coordinate[n]
        sidechain_mass += mass
    if sidechain_mass > 0:
      sidechain_CoM /= sidechain_mass
    else:  # Usually case of Glycin (do not have side chain)
      #'''
      #TO CHANGE FOR NEXT NETWORK ITERATION... I used the wrong nitrogen when I rewrote the function...
      if l>0:
        if (0 in atom_id) & (1 in atom_id) & (17 in atom_ids[l-1]):  # If C,N,Calpha are here, place virtual CoM
          # SCoM = 3 * coord of Ca(current residue) - coord of N(last residue) - coord of C(current residue)
          sidechain_CoM = 3 * atom_coordinate[atom_id.index(1)] - atom_coordinates[l-1][atom_ids[l-1].index(17)] - \
                          atom_coordinate[atom_id.index(0)]
        else:
          if verbose:
            print('Warning, pathological amino acid missing side chain and backbone', l)
          sidechain_CoM = atom_coordinate[-1] # coordinate of last heavy atom
      else:
        if verbose:
          print('Warning, pathological amino acid missing side chain and backbone', l)
        sidechain_CoM = atom_coordinate[-1] # coordinate of last heavy atom
      #'''

      # if (0 in atom_id) & (1 in atom_id) & (17 in atom_id):  # If C,N,Calpha are here, place virtual CoM
      #     sidechain_CoM = 3 * atom_coordinate[atom_id.index(1)] - atom_coordinate[atom_id.index(17)] - \
      #                     atom_coordinate[atom_id.index(0)]
      # else:
      #     if verbose:
      #         print('Warning, pathological amino acid missing side chain and backbone', l)
      #     sidechain_CoM = atom_coordinate[-1]

    aa_clouds.append(sidechain_CoM)
    next = 1 * count
    count += 1
    # triplet is (coordinate_index of center point, previous point, next point)
    # coordinate_index is index of aa_clouds, that we can use the index to search point coordinate from aa_clouds
    aa_triplets.append((center, previous, next))

  return aa_clouds, aa_triplets

