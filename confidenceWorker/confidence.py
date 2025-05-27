# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
import time

import torch
import torch.nn as nn
from torch import distributed as dist

from confidenceWorker.misc import feat_batch
from confidenceWorker.network.head import  ConfidenceHead

class ConfidenceOne():
    def __init__(self):
        super(ConfidenceOne, self).__init__()
        self.confidence_head = ConfidenceHead()
        self.num_samples=5

    def getOnnxModel(self ,batch,embeddings,positions, save_path, ):
        # batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res
        # target_feat=target_feat.to(device=target_feat.device)
        output_names = ["predicted_lddt","predicted_experimentally_resolved","full_pde","average_pde",
            "full_pae","tmscore_adjusted_pae_global","tmscore_adjusted_pae_interface"]
        ordered_keys=['dense_atom_positions','single', 'pair', 'target_feat',
                        'seq_mask','ta_to_pb_gather_idxs','ta_to_pb_gather_mask ','asym_id',
                        ]
        kwarg_inputs = {
            'dense_atom_positions':positions,
            'single': embeddings['single'].to(dtype=torch.float32),
            'pair': embeddings['pair'].to(dtype=torch.float32),
            'target_feat': embeddings['target_feat'].to(dtype=torch.float32),
            'seq_mask': batch.token_features.mask,
            'ta_to_pb_gather_idxs' :batch.pseudo_beta_info.token_atoms_to_pseudo_beta.gather_idxs,
            'ta_to_pb_gather_mask ':batch.pseudo_beta_info.token_atoms_to_pseudo_beta.gather_mask,
            'asym_id': batch.token_features.asym_id,
        }
        ordered_inputs = tuple(kwarg_inputs[key] for key in ordered_keys)
        opset_version=22
        print("opset: ", opset_version)
        print("start to save confidence_head to: ", save_path)
        seq_len = torch.export.Dim('seq_len', min=1, max=100000)
        torch.onnx.export(self.confidence_head,
                      ordered_inputs, f=save_path, dynamo=True,
                      input_names=ordered_keys,
                      output_names=output_names,
                      optimize=True,
                      opset_version=opset_version,
                      export_params=True,
                      dynamic_shapes={
                          'dense_atom_positions':{0:seq_len},
                          'single': {0: seq_len},
                          'pair': {0: seq_len, 1: seq_len},
                          'target_feat': {0: seq_len},
                          'seq_mask': {0: seq_len},
                          'ta_to_pb_gather_idxs': {0: seq_len},
                          'ta_to_pb_gather_mask': {0: seq_len},
                          'asym_id': {0: seq_len},
                          # 'seq_mask': {0: seq_len},
                      }
                          )
    def forward(self, batch
            ,embeddings,positions,
                ) :
        (predicted_lddt,predicted_experimentally_resolved,full_pde,average_pde,
            full_pae,tmscore_adjusted_pae_global,tmscore_adjusted_pae_interface)=self.confidence_head(
                dense_atom_positions=positions,
                # embeddings=embeddings,
                single=embeddings['single'],
                pair=embeddings['pair'],
                target_feat=embeddings['target_feat'],
                seq_mask=batch.token_features.mask,
                ta_to_pb_gather_idxs=batch.pseudo_beta_info.token_atoms_to_pseudo_beta.gather_idxs,
                ta_to_pb_gather_mask=batch.pseudo_beta_info.token_atoms_to_pseudo_beta.gather_mask,
                asym_id=batch.token_features.asym_id
            )

        return (predicted_lddt,predicted_experimentally_resolved,full_pde,average_pde,
            full_pae,tmscore_adjusted_pae_global,tmscore_adjusted_pae_interface)
