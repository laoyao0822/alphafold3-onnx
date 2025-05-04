from evoWorker.network.template import TemplateEmbedding
from evoWorker.network.pairformer import EvoformerBlock, PairformerBlock
import torch
import torch.nn as nn
from evoWorker.network.dot_product_attention import get_attn_mask
class Evoformer(nn.Module):
    def __init__(self, msa_channel: int = 64):
        super(Evoformer, self).__init__()

        self.msa_channel = msa_channel
        self.msa_stack_num_layer = 4
        self.pairformer_num_layer = 48
        self.num_msa = 1024

        self.seq_channel = 384
        self.pair_channel = 128
        self.c_target_feat = 447

        self.c_rel_feat = 139

        self.template_embedding = TemplateEmbedding(
            pair_channel=self.pair_channel)

        self.msa_stack = nn.ModuleList(
            [EvoformerBlock() for _ in range(self.msa_stack_num_layer)])

        self.trunk_pairformer = nn.ModuleList(
            [PairformerBlock() for _ in range(self.pairformer_num_layer)])

    def _embed_template_pair(
            self, num_res,attn_mask_4):
        """Embeds Templates and merges into pair activations."""
        # templates = batch.templates
        # asym_id = batch.token_features.asym_id
        self.template_embedding(num_res,attn_mask_4)

    def _embed_process_msa(
            self, num_res,attn_mask_4
    ):
        """Processes MSA and returns updated pair activations."""

        # Evoformer MSA stack.
        for msa_block in self.msa_stack:
            msa_block(num_res,attn_mask_4)

    def forward(
            self, num_res,attn_mask_4
    ):
        # batch = feat_batch.Batch.from_data_dict(batch)


        # T1 = time.time()
        self._embed_template_pair(num_res,attn_mask_4)
        # T2 = time.time()
        # print(f"pair embedding time: {T2 - T1}")

        self._embed_process_msa(
            num_res,attn_mask_4
        )

        for pairformer_b in self.trunk_pairformer:
            pairformer_b(num_res,attn_mask_4)

        return

class EvoFormerOne(nn.Module):

    def __init__(self, num_recycles: int = 10, num_samples: int = 5, diffusion_steps: int = 200):
        super(EvoFormerOne, self).__init__()

        self.num_recycles = num_recycles
        # self.num_recycles = 1

        self.num_samples = num_samples

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer = Evoformer()
    def forward(self,num_tokens, attn_mask_4) :
        # seq_mask = batch.token_features.mask
        # num_tokens = seq_mask.shape[0]
        # attn_mask_seq = get_attn_mask(mask=seq_mask, dtype=torch.float32, device='cpu', num_heads=16,
        #                               seq_len=num_tokens, batch_size=1)
        # pair_mask = seq_mask[:, None] * seq_mask[None, :]
        # attn_mask_4 = get_attn_mask(mask=pair_mask, dtype=target_feat.dtype,
        #                             device='cpu',
        #                             batch_size=num_tokens,
        #                             num_heads=4, seq_len=num_tokens).contiguous()
        # pair_mask = pair_mask.to(dtype=torch.float32).contiguous()
        for i in range(self.num_recycles+1):
            self.evoformer(num_tokens, attn_mask_4)
            print('success one')

