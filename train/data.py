"""Dataset and collator for draft training.

Each record becomes [prompt tokens | completion tokens]. Loss is taken on the
completion only. The teacher's top-k distribution is carried alongside, laid
out in the same [B, L, K] frame as the input so the usual next-token shift
applies to it too -- getting that shift wrong is the classic KD bug and it
shows up as a loss that trains but a draft model that never gets accepted.
"""
import json

import torch
from torch.utils.data import Dataset


class TraceDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len=2048, topk=8, limit=0):
        self.tok = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.topk = int(topk)
        self.recs = []
        self.skipped_too_long = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p_ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
                c_ids = list(r["token_ids"])
                if len(p_ids) + len(c_ids) > self.max_seq_len:
                    # Truncating would cut the reasoning mid-derivation and teach
                    # the draft to stop early; drop it instead.
                    self.skipped_too_long += 1
                    continue
                self.recs.append((p_ids, c_ids, r.get("tk_ids"), r.get("tk_lps")))
                if limit and len(self.recs) >= limit:
                    break

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return self.recs[i]


def make_collate(pad_id, topk=8):
    def collate(batch):
        L = max(len(p) + len(c) for p, c, _, _ in batch)
        B, K = len(batch), topk
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        loss_mask = torch.zeros(B, L, dtype=torch.bool)
        attn = torch.zeros(B, L, dtype=torch.long)
        tk_ids = torch.zeros(B, L, K, dtype=torch.long)
        tk_lp = torch.full((B, L, K), -1e4, dtype=torch.float)
        tk_ok = torch.zeros(B, L, K, dtype=torch.bool)
        for b, (p, c, ti, tl) in enumerate(batch):
            n, P = len(p) + len(c), len(p)
            input_ids[b, :n] = torch.tensor(p + c)
            attn[b, :n] = 1
            loss_mask[b, P:n] = True             # completion only
            if ti:
                for t in range(min(len(ti), len(c))):
                    ids, lps = ti[t][:K], tl[t][:K]
                    m = len(ids)
                    tk_ids[b, P + t, :m] = torch.tensor(ids, dtype=torch.long)
                    tk_lp[b, P + t, :m] = torch.tensor(lps, dtype=torch.float)
                    tk_ok[b, P + t, :m] = True
        return {"input_ids": input_ids, "attention_mask": attn, "loss_mask": loss_mask,
                "tk_ids": tk_ids, "tk_lp": tk_lp, "tk_ok": tk_ok}
    return collate
