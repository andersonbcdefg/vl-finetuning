import re
from typing import Any

import torch

MAX_PIXELS = 1024 * 1024


def build_ntl_index(tokenizer: Any, vocab_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    number_re = re.compile(r"^[+-]?\d+$")  # lenient integer regex
    num_ids, num_vals = [], []

    for i in range(vocab_size):
        s = tokenizer.decode([i]).strip()
        if number_re.fullmatch(s):
            num_ids.append(i)
            num_vals.append(float(s))

    num_ids = torch.tensor(num_ids, dtype=torch.long)  # (N,)
    num_vals = torch.tensor(num_vals, dtype=torch.float32)  # (N,)

    # id → value lookup for fast scatter/gather
    token_id_to_val = torch.full((vocab_size,), float("nan"), dtype=torch.float32)
    token_id_to_val[num_ids] = num_vals

    return num_ids.to(device), num_vals.to(device), token_id_to_val.to(device)

@torch.compile
def compute_ntl_loss(num_ids, num_vals, token_id_to_val, last_hidden, labels, lm_head):
    h_pred = last_hidden[:, :-1]
    lab_t  = labels[:, 1:]

    num_pos_mask = torch.isfinite(token_id_to_val[lab_t])          # (B, L-1)
    h_num        = h_pred[num_pos_mask]                            # (K, H)
    tgt_ids      = lab_t[num_pos_mask]                             # (K,)
    tgt_val      = token_id_to_val[tgt_ids]                        # (K,)

    partial_lm   = lm_head.index_select(0, num_ids)                # (N, H)
    logits_num   = h_num @ partial_lm.T                            # (K, N)
    exp_val      = torch.softmax(logits_num, -1).matmul(num_vals)  # (K,)

    # if no numeric positions K==0, tensors are empty and MSE returns 0
    return torch.nn.functional.mse_loss(exp_val, tgt_val, reduction="mean")


    # h_pred = last_hidden[:, :-1, :]  # (B, L-1, H)
    # lab_t = labels[:, 1:]  # (B, L-1)

    # # Mask time‑steps whose GT token is numeric
    # num_pos_mask = token_id_to_val[lab_t].isfinite()  # boolean (B, L-1)
    # if num_pos_mask.any():
    #     h_num = h_pred[num_pos_mask]  # (K, H)
    #     tgt_ids = lab_t[num_pos_mask]  # (K,)
    #     tgt_val = token_id_to_val[tgt_ids]  # (K,)

    #     # Logits restricted to numeric tokens only
    #     partial_lm_head = torch.index_select(
    #         lm_head, 0, num_ids
    #     )  # avoids advanced‑index view

    #     ntl_loss = _compute_ntl_loss(h_num, partial_lm_head, num_vals)

    # else:
    #     ntl_loss = torch.tensor(0.0, device=last_hidden.device, dtype=last_hidden.dtype)

    # return ntl_loss
