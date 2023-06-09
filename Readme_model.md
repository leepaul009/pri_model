

### hox model:

```
GraphNet(
  (atom_emb_layer): Embedding(39, 128)
  (atom_level_sub_graph): SubGraph(
    (layer_0): MLP(
      (linear): Linear(in_features=38, out_features=64, bias=True)
      (layer_norm): LayerNorm()
    )
    (layers): ModuleList(
      (0): GlobalGraph(
        (query): Linear(in_features=64, out_features=64, bias=True)
        (key): Linear(in_features=64, out_features=64, bias=True)
        (value): Linear(in_features=64, out_features=64, bias=True)
      )
      (1): GlobalGraph(
        (query): Linear(in_features=64, out_features=64, bias=True)
        (key): Linear(in_features=64, out_features=64, bias=True)
        (value): Linear(in_features=64, out_features=64, bias=True)
      )
      (2): GlobalGraph(
        (query): Linear(in_features=64, out_features=64, bias=True)
        (key): Linear(in_features=64, out_features=64, bias=True)
        (value): Linear(in_features=64, out_features=64, bias=True)
      )
    )
    (layers_2): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
      (2): LayerNorm()
    )
    (layer_0_again): MLP(
      (linear): Linear(in_features=64, out_features=64, bias=True)
      (layer_norm): LayerNorm()
    )
    (atom_emb_layer): Embedding(39, 38)
  )
  (global_graph): GlobalGraphRes(
    (global_graph): GlobalGraph(
      (query): Linear(in_features=128, out_features=64, bias=True)
      (key): Linear(in_features=128, out_features=64, bias=True)
      (value): Linear(in_features=128, out_features=64, bias=True)
    )
    (global_graph2): GlobalGraph(
      (query): Linear(in_features=128, out_features=64, bias=True)
      (key): Linear(in_features=128, out_features=64, bias=True)
      (value): Linear(in_features=128, out_features=64, bias=True)
    )
  )
  (laneGCN_A2L): CrossAttention(
    (query): Linear(in_features=64, out_features=64, bias=True)
    (key): Linear(in_features=64, out_features=64, bias=True)
    (value): Linear(in_features=64, out_features=64, bias=True)
  )
  (laneGCN_L2L): GlobalGraphRes(
    (global_graph): GlobalGraph(
      (query): Linear(in_features=64, out_features=32, bias=True)
      (key): Linear(in_features=64, out_features=32, bias=True)
      (value): Linear(in_features=64, out_features=32, bias=True)
    )
    (global_graph2): GlobalGraph(
      (query): Linear(in_features=64, out_features=32, bias=True)
      (key): Linear(in_features=64, out_features=32, bias=True)
      (value): Linear(in_features=64, out_features=32, bias=True)
    )
  )
  (laneGCN_L2A): CrossAttention(
    (query): Linear(in_features=64, out_features=64, bias=True)
    (key): Linear(in_features=64, out_features=64, bias=True)
    (value): Linear(in_features=64, out_features=64, bias=True)
  )
  (laneGCN_A2A): GlobalGraphRes(
    (global_graph): GlobalGraph(
      (query): Linear(in_features=64, out_features=32, bias=True)
      (key): Linear(in_features=64, out_features=32, bias=True)
      (value): Linear(in_features=64, out_features=32, bias=True)
    )
    (global_graph2): GlobalGraph(
      (query): Linear(in_features=64, out_features=32, bias=True)
      (key): Linear(in_features=64, out_features=32, bias=True)
      (value): Linear(in_features=64, out_features=32, bias=True)
    )
  )
  (aa_embeding_layer): TimeDistributed(
    (module): Linear(in_features=20, out_features=64, bias=False)
  )
  (kmers_net): KMersNet(
    (conv): Conv2d(
      (conv): Conv2d(1, 8, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2))
      (bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (res2d): ResidualBlock2D(
      (c1): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (b1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (c2): Conv2d(8, 16, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False)
      (b2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (c3): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (b3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (fc): Linear(in_features=128, out_features=64, bias=True)
    (relu): ReLU(inplace=True)
  )
  (reg_head): Linear(in_features=256, out_features=128, bias=False)
  (reg_pred): Linear(in_features=128, out_features=1, bias=False)
  (loss): PredLoss(
    (reg_loss): SmoothL1Loss()
  )
)
```