"""MPS (Apple GPU) path for the sparse connectome core.

PyTorch has no sparse-CSR kernels on the MPS backend, so torch.sparse.mm raises
NotImplementedError there. This module installs a drop-in replacement for
``_EdgeSparseMM`` that evaluates exactly the same mathematics with dense
gather + ``index_add_`` (a chunked scatter-add), which MPS supports.

Forward:   Y[i] = sum_{edges e: i <- j} values[e] * X[j]
Backward:  dvalues[e] = <dY[i], X[j]>          (exact, at measured edges only)
           dX[j]      = sum_{e: i <- j} values[e] * dY[i]

Nothing in the model, its parameters, or its topology changes. On CPU/CUDA the
original sparse path is left untouched, so this file only matters when the
model lives on ``mps``.
"""
import torch

import flyhard.connectome as _c

CHUNK = 1 << 21  # edges per chunk; bounds temporary memory (~CHUNK*batch*4 bytes)


class _EdgeSparseMMGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, crow, col, rows, state):
        ctx.save_for_backward(values, crow, col, rows, state)
        out = torch.zeros_like(state)
        for start in range(0, len(values), CHUNK):
            end = min(start + CHUNK, len(values))
            out.index_add_(0, rows[start:end], values[start:end, None] * state[col[start:end]])
        return out

    @staticmethod
    def backward(ctx, output_grad):
        values, crow, col, rows, state = ctx.saved_tensors
        value_grad = None
        if ctx.needs_input_grad[0]:
            value_grad = torch.empty_like(values)
            for start in range(0, len(values), CHUNK):
                end = min(start + CHUNK, len(values))
                value_grad[start:end] = (output_grad[rows[start:end]] * state[col[start:end]]).sum(dim=1)
        state_grad = None
        if ctx.needs_input_grad[4]:
            state_grad = torch.zeros_like(state)
            for start in range(0, len(values), CHUNK):
                end = min(start + CHUNK, len(values))
                state_grad.index_add_(0, col[start:end], values[start:end, None] * output_grad[rows[start:end]])
        return value_grad, None, None, None, state_grad


_original = _c._EdgeSparseMM


class _Dispatch:
    """Route to the gather kernel on MPS and to the original kernel elsewhere."""
    @staticmethod
    def apply(values, crow, col, rows, state):
        if values.device.type == 'mps':
            return _EdgeSparseMMGather.apply(values, crow, col, rows, state)
        return _original.apply(values, crow, col, rows, state)


def install():
    _c._EdgeSparseMM = _Dispatch


install()
