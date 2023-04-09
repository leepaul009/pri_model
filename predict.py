import os
import numpy as np
import argparse
from preprocessing import PDBio
from preprocessing import pipelines
from preprocessing import PDB_processing
from utilities.paths import structures_folder





if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('input',  type=str, help='')
  parser.add_argument('--noMSA', dest='use_MSA', action='store_const',
                  const = False, default = True, help = '')
  args = parser.parse_args()
  input = args.input

  query_pdbs, query_chain_ids = PDBio.parse_str(input)
  pipeline = pipelines.ScanNetPipeline(
    with_aa=True,
    with_atom=True,
    aa_features='sequence',
    atom_features='valency',
    aa_frames='triplet_sidechain',
    Beff=500,
  )

  query_pdbs = [query_pdbs]
  query_chain_ids = [query_chain_ids]

  pdb_file_locations = []
  for pdb_id in query_pdbs:
    location, chain = PDBio.getPDB(pdb_id,biounit=True,
      structures_folder=structures_folder)
    if os.path.exists(location):
      pdb_file_locations.append(location)
  
  query_chain_objs = []
  for i in range(len(pdb_file_locations)):
    _, chain_objs = PDBio.load_chains(
      chain_ids= query_chain_ids[i], file=pdb_file_locations[i])
    query_chain_objs.append(chain_objs)

  query_sequences = [[PDB_processing.process_chain(chain_obj)[0]
    for chain_obj in chain_objs] for chain_objs in query_chain_objs]

  query_residue_ids =[]
  for i, chain_objs in enumerate(query_chain_objs):
    if chain_objs is not None:
      residue_ids =  PDB_processing.get_PDB_indices(chain_objs, 
        return_chain=True, return_model=True)
      
      query_residue_ids.append(residue_ids)

  #



  