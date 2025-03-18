from typing import Optional
import torch



def dot_product_attention_torch(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    # if mask is not None:
    #     print("mask is not None")
    # if bias is not None:
    #     print("bias is not None")


    scaling = q.size(-1) ** -0.5
    q = q * scaling
    logits = torch.matmul(q, k.transpose(-1, -2))
    # if mask is not None and bias is not None:
    #     print(logits.shape,mask.shape,bias.shape)
    # if mask is not None and bias is  None:
    #     print("mask is not None and bias is None")

    if bias is not None:
        logits += bias

    if mask is not None:
        if mask.dim() == 1:
            mask = mask[None, None, None, :].to(dtype=torch.bool)
        elif mask.dim() == 2:
            mask = mask[:, None, None, :].to(dtype=torch.bool)
        logits.masked_fill_(~mask, -torch.finfo(logits.dtype).max)



    weights = torch.softmax(logits, dim=-1)

    return torch.matmul(weights, v)

