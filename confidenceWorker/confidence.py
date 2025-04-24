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

    def forward(self, batch: dict[str, torch.Tensor]
                ,embeddings,positions
                ) :
        batch = feat_batch.Batch.from_data_dict(batch)

        sample_mask = batch.predicted_structure_info.atom_mask
        # samples = self._sample_diffusion(batch, embeddings)
        final_dense_atom_mask = torch.tile(sample_mask[None], (self.num_samples, 1, 1))
        samples={'atom_positions': positions, 'mask': final_dense_atom_mask}
        time2=time.time()
        confidence_output_per_sample = []
        # print("positions shape:",positions.shape)
        for sample_dense_atom_position in samples['atom_positions']:
            # print("sample_dense_atom_position", sample_dense_atom_position.shape)
            (predicted_lddt,predicted_experimentally_resolved,full_pde,average_pde,
             full_pae,tmscore_adjusted_pae_global,tmscore_adjusted_pae_interface)=self.confidence_head(
                dense_atom_positions=sample_dense_atom_position,
                # embeddings=embeddings,
                single=embeddings['single'],
                pair=embeddings['pair'],
                target_feat=embeddings['target_feat'],
                seq_mask=batch.token_features.mask,
                # token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                ta_to_pb_gather_idxs=batch.pseudo_beta_info.token_atoms_to_pseudo_beta.gather_idxs,
                ta_to_pb_gather_mask=batch.pseudo_beta_info.token_atoms_to_pseudo_beta.gather_mask,
                asym_id=batch.token_features.asym_id
            )
            # print("predicted_lddt:",predicted_lddt.shape)
            # print("predicted_experimentally_resolved:",predicted_experimentally_resolved.shape)
            # print("full_pde:",full_pde.shape)
            # print("average_pde:",average_pde.shape)
            # print("full_pae:",full_pae.shape)
            # print("tmscore_adjusted_pae_global:",tmscore_adjusted_pae_global.shape)
            # print("tmscore_adjusted_pae_interface:",tmscore_adjusted_pae_interface.shape)
            confidence_putput={
                'predicted_lddt': predicted_lddt,
                'predicted_experimentally_resolved': predicted_experimentally_resolved,
                'full_pde': full_pde,
                'average_pde': average_pde,
                'full_pae':full_pae,
                'tmscore_adjusted_pae_global': tmscore_adjusted_pae_global,
                'tmscore_adjusted_pae_interface': tmscore_adjusted_pae_interface
            }
            confidence_output_per_sample.append(confidence_putput)
        print("confidence_head cost time:",time.time()-time2)
        # print("confidence_output_per_sample:",confidence_output_per_sample[0].shape)
        confidence_output = {}
        for key in confidence_output_per_sample[0]:
            confidence_output[key] = torch.stack(
                [sample[key] for sample in confidence_output_per_sample], dim=0)
        return confidence_output,samples
