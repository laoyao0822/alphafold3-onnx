import torch
import torch.nn as nn
from torch.nn import LayerNorm

from diffusionWorker2.network.diffusion_transformer import  DiffusionTransition
from diffusionWorker2.premodel.pre_ac_encoder import AtomCrossAttEncoder
from diffusionWorker2.premodel.pre_diffusionTransformer import DiffusionTransformer

class DiffusionHead(nn.Module):
    def __init__(self):
        super(DiffusionHead, self).__init__()
        self.pair_channel = 128

        self.c_pair_cond_initial = 267
        self.seq_channel = 384

        self.c_pair_cond_initial = 267
        self.pair_cond_initial_norm = LayerNorm(
            self.c_pair_cond_initial, bias=False)
        self.pair_cond_initial_projection = nn.Linear(
            self.c_pair_cond_initial, self.pair_channel, bias=False)

        self.pair_transition_0 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)
        self.pair_transition_1 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)


        self.c_single_cond_initial = 831

        self.single_cond_initial_projection = nn.Linear(
            self.c_single_cond_initial, self.seq_channel, bias=False)
        self.single_cond_initial_norm = LayerNorm(
            self.c_single_cond_initial, bias=False)

        self.atom_cross_att_encoder = AtomCrossAttEncoder()
        self.transformer = DiffusionTransformer()


    def _conditioning(
        self,
        rel_features,
        single, pair, target_feat,
    ) :
        features_2d = torch.concatenate([pair, rel_features], dim=-1)

        pair_cond = self.pair_cond_initial_projection(
            self.pair_cond_initial_norm(features_2d)
        )
        pair_cond += self.pair_transition_0(pair_cond)
        pair_cond += self.pair_transition_1(pair_cond)

        # target_feat = embeddings['target_feat']
        features_1d = torch.concatenate(
            [single, target_feat], dim=-1)

        single_cond = self.single_cond_initial_projection(
            self.single_cond_initial_norm(features_1d))
        return pair_cond, single_cond

    def forward(self,
                rel_features,single,pair,target_feat,
                ref_ops, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid,

                queries_mask,
                acat_atoms_to_q_gather_idxs,
                acat_atoms_to_q_gather_mask,

                acat_q_to_k_gather_idxs,
                acat_q_to_k_gather_mask,

                acat_t_to_q_gather_idxs,
                acat_t_to_q_gather_mask,

                acat_t_to_k_gather_idxs,
                acat_t_to_k_gather_mask,
                ):
        pair_cond,single_cond = self._conditioning(rel_features,single,pair,target_feat)
        pair_logits_cat=self.transformer(pair_cond)
        print('pair_logits_cat', pair_logits_cat.shape)
        queries_single_cond,pair_act,keys_mask,keys_single_cond=self.atom_cross_att_encoder(
            trunk_single_cond=single,trunk_pair_cond=pair_cond.clone(),
            ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,

            queries_mask=queries_mask,
            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,

            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,

            acat_t_to_q_gather_idxs=acat_t_to_q_gather_idxs,
            acat_t_to_q_gather_mask=acat_t_to_q_gather_mask,

            acat_t_to_k_gather_idxs=acat_t_to_k_gather_idxs,
            acat_t_to_k_gather_mask=acat_t_to_k_gather_mask,
          )
        return  single_cond,pair_cond,queries_single_cond,pair_act,keys_mask,keys_single_cond,pair_logits_cat
