
import torch

from modeling.backbone.globalnet import GlobalNet

from modeling.esm.data import Alphabet
from modeling.dbert.data import Alphabet as DAlphabet
from modeling.batch_convert.batch_convert import BasicBatchConvert



def freeze_esm_dbert_layers(model, last_freeze_layer = 0):
  if last_freeze_layer == -1:
    print("do not freeze layers, layer = {}" .format(last_freeze_layer))
    return model

  print("freeze layers up to {}-th TF layer and embedding layer during training"
        .format(last_freeze_layer))
  ### freeze esm layers
  for param in model.esm2.embed_tokens.parameters():
    param.requires_grad = False
  for i in range(last_freeze_layer + 1):
    for param in model.esm2.layers[i].parameters():
      param.requires_grad = False

  ### freeze dbert layers
  for param in model.dbm.embeddings.parameters():
    param.requires_grad = False
  for i in range(last_freeze_layer + 1):
    for param in model.dbm.encoder.layer[i].parameters():
      param.requires_grad = False

  ### show result
  for name, param in model.named_parameters():
    print(name, param.requires_grad)
  
  return model
  
  

def build_esm_dbert_model(args):
  
  path_esm2_cp = "dataset/checkpoints/esm2_t6_8M_UR50D.pt"
  checkpoint = torch.load(path_esm2_cp, map_location=torch.device('cpu'))
  esm2_cfg = checkpoint['cfg_model']
  esm2_state = checkpoint['model']
  # esm2_state.update(checkpoint["regression"])
  
  path_dbm_cp = 'dataset/checkpoints/dnabert5_t12.pt'
  checkpoint = torch.load(path_dbm_cp, map_location=torch.device('cpu'))
  dbm_state = checkpoint['model']

  alphabet = Alphabet.from_architecture("ESM-1b")
  vocab_file = 'dataset/checkpoints/dvocab{}.txt'.format(args.kmers)
  dalphabet = DAlphabet.from_architecture(vocab_file)
  
  
  basic_batch_convert = BasicBatchConvert(alphabet, dalphabet)
  model = GlobalNet(args, esm2_cfg, alphabet, dalphabet)

  def update_state_dict_esm(state_dict):
    state_dict = {'esm2.' + name : param for name, param in state_dict.items()}
    return state_dict
  
  def update_state_dict_dbm(state_dict):
    state_dict = {'dbm.' + name : param for name, param in state_dict.items()}
    return state_dict

  ### only load esm2 checkpoint
  esm2_state = update_state_dict_esm(esm2_state)
  dbm_state  = update_state_dict_dbm(dbm_state)
  esm2_state.update(dbm_state)
  
  model.load_state_dict(esm2_state, strict=False)
  
  model = freeze_esm_dbert_layers(model, last_freeze_layer=args.freeze_layer)
  
  
  
  
  return model, basic_batch_convert