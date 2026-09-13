"""Sparse recurrent rate model with immutable measured adjacency.

Each measured edge has a trainable bounded gain, and every neuron has a
trainable leak rate. E01 uses unsigned transmission as a numerical test;
transmitter/receptor biology is not asserted by this model.
"""
import torch
from torch import nn


MPS_EDGE_CHUNK = 4_194_304
CUDA_EDGE_CHUNK = 8_388_608   # CUDA has the memory for large edge chunks; small chunks are launch-bound


class _EdgeSparseMM(torch.autograd.Function):
    """First-order SpMM derivative evaluated only at measured edges.

    PyTorch 2.8's CSR-value backward allocated a dense N-by-N intermediate
    on the full graph (101.57 GiB). The exact derivative at an edge i<-j is
    dot(dL/dY[i], X[j]). Chunked indexing bounds temporary memory; the state
    derivative remains an ordinary PyTorch sparse matrix multiplication.
    """
    @staticmethod
    def forward(ctx, values, crow, col, rows, state):
        n = len(crow)-1
        matrix = torch.sparse_csr_tensor(crow, col, values, size=(n,n), check_invariants=False)
        ctx.save_for_backward(values, crow, col, rows, state)
        return torch.sparse.mm(matrix, state)

    @staticmethod
    def backward(ctx, output_grad):
        values, crow, col, rows, state = ctx.saved_tensors
        value_grad = torch.empty_like(values) if ctx.needs_input_grad[0] else None
        if value_grad is not None:
            chunk = CUDA_EDGE_CHUNK if values.is_cuda else 262144
            for start in range(0, len(values), chunk):
                end = min(start+chunk, len(values))
                value_grad[start:end] = (
                    output_grad[rows[start:end]] * state[col[start:end]]
                ).sum(dim=1)
        state_grad = None
        if ctx.needs_input_grad[4]:
            n = len(crow)-1
            if values.is_cuda:
                # cuSPARSE re-converts a transposed CSR on every call (~7x the forward cost), so keep the
                # transposed structure once and only permute the current values into it.
                crow_t, col_t, perm = _transposed_structure(crow, col, rows, n)
                matrix_t = torch.sparse_csr_tensor(crow_t, col_t, values[perm], size=(n,n), check_invariants=False)
                state_grad = torch.sparse.mm(matrix_t, output_grad)
            else:
                matrix = torch.sparse_csr_tensor(crow, col, values, size=(n,n), check_invariants=False)
                state_grad = torch.sparse.mm(matrix.transpose(0,1), output_grad)
        return value_grad, None, None, None, state_grad


_TRANSPOSE_CACHE = {}


def _transposed_structure(crow, col, rows, n):
    """CSR structure of the transposed graph plus the permutation from edge order to transposed order."""
    key = (col.data_ptr(), crow.data_ptr(), str(col.device))
    if key not in _TRANSPOSE_CACHE:
        perm = torch.argsort(col, stable=True)            # edges grouped by presynaptic neuron, rows ascending inside
        counts = torch.bincount(col, minlength=n)
        crow_t = torch.zeros(n+1, dtype=crow.dtype, device=col.device); crow_t[1:] = torch.cumsum(counts, 0)
        _TRANSPOSE_CACHE[key] = (crow_t, rows[perm], perm)
    return _TRANSPOSE_CACHE[key]


class _EdgeGatherMM(torch.autograd.Function):
    """Same product, written with gather and scatter-add instead of CSR.

    Apple's MPS backend has no compressed-sparse tensor, so `torch.sparse.mm`
    raises there. Accumulating `values[e] * state[col[e]]` into `rows[e]` is
    the same arithmetic on supported ops; chunking bounds the E-by-batch
    temporaries. Row order inside the CSR is canonical, so the summation
    order matches and results agree with the CSR path to float tolerance.
    """
    @staticmethod
    def forward(ctx, values, crow, col, rows, state):
        ctx.save_for_backward(values, col, rows, state)
        out = torch.zeros(len(crow)-1, state.shape[1], device=state.device, dtype=state.dtype)
        for start in range(0, len(values), MPS_EDGE_CHUNK):
            end = min(start+MPS_EDGE_CHUNK, len(values))
            out.index_add_(0, rows[start:end], values[start:end, None]*state[col[start:end]])
        return out

    @staticmethod
    def backward(ctx, output_grad):
        values, col, rows, state = ctx.saved_tensors
        value_grad = torch.empty_like(values) if ctx.needs_input_grad[0] else None
        state_grad = torch.zeros_like(state) if ctx.needs_input_grad[4] else None
        for start in range(0, len(values), MPS_EDGE_CHUNK):
            end = min(start+MPS_EDGE_CHUNK, len(values))
            gathered = output_grad[rows[start:end]]
            if value_grad is not None:
                value_grad[start:end] = (gathered * state[col[start:end]]).sum(dim=1)
            if state_grad is not None:
                state_grad.index_add_(0, col[start:end], gathered * values[start:end, None])
        return value_grad, None, None, None, state_grad


def edge_matmul(values, crow, col, rows, state):
    """Pick the kernel the device supports; the maths is identical."""
    if state.device.type == 'mps':
        return _EdgeGatherMM.apply(values, crow, col, rows, state)
    return _EdgeSparseMM.apply(values, crow, col, rows, state)


class SparseConnectome(nn.Module):
    def __init__(self, crow, col, counts, *, edge_init=0.0, leak_init=0.0):
        super().__init__()
        self.n = len(crow) - 1
        self.register_buffer("crow", torch.as_tensor(crow, dtype=torch.int64))
        self.register_buffer("col", torch.as_tensor(col, dtype=torch.int64))
        counts = torch.as_tensor(counts, dtype=torch.float32)
        rows = torch.repeat_interleave(torch.arange(self.n), torch.diff(self.crow))
        self.register_buffer("rows", rows)
        totals = torch.zeros(self.n).index_add_(0, rows, counts)
        self.register_buffer("base", counts / totals[rows].clamp_min(1))
        self.edge_gain = nn.Parameter(torch.full_like(counts, float(edge_init)))
        self.leak = nn.Parameter(torch.full((self.n,), float(leak_init)))

    def edge_values(self):
        return self.base * (0.05 + 0.90 * torch.sigmoid(self.edge_gain))

    def matrix(self, values=None):
        if values is None:
            values = self.edge_values()
        return torch.sparse_csr_tensor(self.crow, self.col, values, size=(self.n, self.n), check_invariants=False)

    def forward(self, state, steps=1, drive=None):
        """State shape [neurons, batch]; row=postsynaptic, col=presynaptic.

        An optional external drive must already be mapped to declared input
        neurons by the experiment. There is no direct input-to-output bypass.
        """
        values = self.edge_values()
        leak = (0.05 + 0.90 * torch.sigmoid(self.leak))[:, None]
        for _ in range(steps):
            signal = edge_matmul(values, self.crow, self.col, self.rows, state)
            if drive is not None:
                signal = signal + drive
            state = (1 - leak) * state + leak * torch.tanh(signal)
        return state
