import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import sys


import os, sys
ACT_THRESHOLD = os.environ.get('ACT_THRESHOLD')
if ACT_THRESHOLD is None:
    ACT_THRESHOLD = 0.999
else:
    ACT_THRESHOLD = float(ACT_THRESHOLD)
    print('-' * 10, file=sys.stderr)
    print("Threshold modified.", file=sys.stderr)
    print("ACT_THRESHOLD=", ACT_THRESHOLD, file=sys.stderr)
    print('-' * 10, file=sys.stderr)



class StickBreakingACT(nn.Module):
    def __init__(self, hidden_size, threshold=0.999):
        super(StickBreakingACT, self).__init__()
        self._gate = nn.Sequential(
            nn.Linear(hidden_size, 2, bias=False)
        )
        nn.init.zeros_(self._gate[-1].weight)
        self.threshold = threshold

    def gate(self, h):
        logits = self._gate(h)
        if self.training:
            logits = logits + torch.randn_like(logits)
        return F.log_softmax(logits, dim=-1)

    def update_halting(self, curr_log_g, prev_out, prev_state):
        if prev_state is None:
            step = 0
            curr_log_halt = curr_log_g
            g = torch.exp(curr_log_halt[..., 1])
            curr_acc_g = g
            curr_acc_h = g[..., None] * prev_out
            curr_expstep = 0.
        else:
            (prev_halt_mask, prev_log_never_halt,
             prev_acc_h, prev_acc_g, prev_step, prev_acc_expstep) = prev_state
            step = prev_step + 1
            curr_log_halt = prev_log_never_halt[..., None] + curr_log_g
            g = torch.exp(curr_log_halt[..., 1])
            g = g.masked_fill(prev_halt_mask, 0.)
            curr_acc_g = prev_acc_g + g
            curr_acc_h = prev_acc_h + g[..., None] * prev_out
            curr_expstep = prev_acc_expstep + g * step

        halt_mask = curr_acc_g >= self.threshold
        curr_state = (halt_mask, curr_log_halt[..., 0],
                      curr_acc_h, curr_acc_g, step, curr_expstep)

        return curr_state, halt_mask

    def forward(self, prev_state, prev_out):
        log_g = self.gate(prev_out)
        return self.update_halting(log_g, prev_out, prev_state)

    def halt_gating(self, curr_state, curr_h):
        halt_mask, _, curr_acc_h, curr_acc_g, step, curr_expstep = curr_state
        soft_halted_h = curr_acc_h + (1 - curr_acc_g)[..., None] * curr_h
        expstep = curr_expstep + (step + 1) * (1 - curr_acc_g)
        if halt_mask.any():
            soft_halted_h.masked_scatter_(halt_mask[..., None], curr_acc_h[halt_mask])
        return soft_halted_h, expstep


class ACTWrapper(nn.Module):
    def __init__(self, mod, threshold=ACT_THRESHOLD, halting_dropout=0):
        super(ACTWrapper, self).__init__()
        self._gate = nn.Sequential(
            nn.Linear(mod.embed_dim, mod.embed_dim),
            nn.GELU(),
            nn.Dropout(halting_dropout),
            nn.Linear(mod.embed_dim, 2, bias=False)
        )
        nn.init.zeros_(self._gate[-1].weight)
        self.threshold = threshold
        self.mod = mod

    def gate(self, h):
        logits = self._gate(h)
        return F.log_softmax(logits, dim=-1)

    def update_halting(self, log_g, log_never_halt):
        log_halt = log_never_halt[..., None] + log_g
        log_never_halt = log_halt[..., 0]
        p = torch.exp(log_halt[..., 1])
        return p, log_never_halt

    def forward(self, prev_act_state, prev_h, self_attn_input, pad_mask, layer_idx, *args, **kwargs):
        if prev_act_state is None:
            log_never_halt = acc_expect_depth = \
                torch.zeros_like(prev_h[..., 0])
            acc_h = torch.zeros_like(prev_h)
            i = 0
            p_never_halt = pad_mask
        else:
            (i, log_never_halt, acc_h, acc_expect_depth) = prev_act_state
            log_g = self.gate(prev_h)
            p, log_never_halt = self.update_halting(log_g, log_never_halt)
            acc_h = acc_h + p[..., None] * prev_h
            acc_expect_depth = acc_expect_depth + i * p
            p_never_halt = log_never_halt.exp()
            p_never_halt = p_never_halt.masked_fill((p_never_halt < (1 - self.threshold)), 0) * pad_mask
            p_never_halt = p_never_halt.contiguous()
            i = i + 1

        curr_act_state = (i, log_never_halt, acc_h, acc_expect_depth)
        outputs = self.mod(prev_h, self_attn_input, p_never_halt, layer_idx, *args, **kwargs)

        curr_h = outputs[0]
        if prev_act_state is not None:
            self_attn_input = torch.where(
                p_never_halt[..., None] < (1 - self.threshold),
                self_attn_input,
                (acc_h + p_never_halt[..., None] * curr_h).type_as(self_attn_input)
            )
            act_loss = (acc_expect_depth + p_never_halt * i) * pad_mask
            act_loss = act_loss.sum() / pad_mask.sum()
        else:
            self_attn_input = curr_h
            act_loss = 0

        return curr_act_state, outputs, self_attn_input, act_loss


if __name__ == "__main__":

    hidden_size = 10
    length = 8
    batch_size = 1
    class Dummy(nn.Module):
        def __init__(self):
            super(Dummy, self).__init__()
            pass
        def forward(self, x, soft_halt_x, halt_mask, layer_idx):
            curr_h = torch.randn(batch_size, length, hidden_size)
            return curr_h,

    layer = ACTWrapper(Dummy(), hidden_size)
    h = torch.randn(batch_size, length, hidden_size)
    soft_halt_x = h
    act_state = layer.init(h)
    for i in range(10):
        prev_h = soft_halt_x
        act_state, outputs, soft_halt_x, act_loss = layer.forward(act_state, h, soft_halt_x, None, i)
        h = soft_halt_x
        halt_mask = act_state[-1]
        torch.set_printoptions(precision=3, linewidth=200, threshold=10000)
        print("----")
        print("same  ", (h[0] == prev_h[0]).all(-1).long())
        print("halted", halt_mask[0].long())
        print("diff  ", torch.abs(h[0] - prev_h[0]).sum(-1))
    print(act_loss)
