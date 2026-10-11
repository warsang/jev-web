"""Export patch for Qwen3.5's Gated DeltaNet.

Problem: transformers' torch_chunk_gated_delta_rule contains
    for i in range(num_chunks):   # num_chunks = ceil(seq_len / 64)
with a sequence-dependent trip count, which forces num_chunks (hence
seq_len) to be static in torch.export and breaks dynamic-seq ONNX export.

Fix: this module provides a mathematically identical implementation whose
second phase uses torch._higher_order_ops.while_loop (dynamic trip count,
supported by the dynamo ONNX exporter) instead of the Python loop.

Implementation notes (all validated against dynamo's HOP tracer):
  * Per-chunk slices use torch.index_select (dynamic index) -- plain
    advanced indexing and view ops (squeeze/reshape/unsqueeze) break HOP
    tracing.
  * All matmuls use torch.einsum with explicit batch dims -- broadcast
    matmul with mismatched batch rank fails, and unsqueeze (a view) is out.
  * The size-1 chunk dim is kept (never squeezed); broadcasting handles it.

Usage (in 02_export.py, before building the model):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm
    import transformers.models.qwen3_5.modular_qwen3_5 as qmm
    from export_patch import patched_torch_chunk_gated_delta_rule as f
    qm.torch_chunk_gated_delta_rule = f
    qmm.torch_chunk_gated_delta_rule = f

Everything before the second phase is copied verbatim from
transformers/models/qwen3_5/modeling_qwen3_5.py::torch_chunk_gated_delta_rule
(transformers 5.18.0).
"""
import torch
import torch.nn.functional as F
from torch._higher_order_ops.while_loop import while_loop


def _l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _is_exporting():
    from transformers.utils.import_utils import is_torchdynamo_exporting
    return is_torchdynamo_exporting()


def patched_torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    # Round the sequence up to a multiple of chunk_size using only
    # division/multiplication (no modulo), so that torch.export can verify
    # total == num_chunks * chunk_size definitionally (the original
    # `(chunk_size - seq % chunk_size) % chunk_size` form generates
    # unprovable guards).
    num_chunks = (sequence_length + chunk_size - 1) // chunk_size
    total_sequence_length = num_chunks * chunk_size
    pad_size = total_sequence_length - sequence_length
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], num_chunks, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], num_chunks, chunk_size)

    strictly_upper_mask = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)

    cum_decay = decay.cumsum(dim=3)

    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    if not _is_exporting():
        new_values = torch.linalg.solve_triangular(
            ut_system, v_beta, upper=False, unitriangular=True
        )
        k_cumdecay = torch.linalg.solve_triangular(
            ut_system, decayed_k_beta, upper=False, unitriangular=True
        )
    else:
        ut_system = -ut_system.tril(-1)
        for i in range(1, chunk_size):
            row = ut_system[..., i, :i].clone()
            sub = ut_system[..., :i, :i].clone()
            ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut_system = ut_system + torch.eye(
            chunk_size, dtype=ut_system.dtype, device=ut_system.device
        )
        new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta

    if initial_state is None:
        last_recurrent_state = torch.zeros(
            recurrent_state_shape, dtype=new_values.dtype, device=new_values.device
        )
    else:
        last_recurrent_state = initial_state.to(new_values.dtype)

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Second phase: sequential scan over chunks via while_loop (dynamic trip
    # count). Mathematically identical to the original Python loop.
    # The accumulator is preallocated to the STATIC max (64 chunks); the
    # while_loop carry must have static shapes, so we slice to num_chunks
    # after the loop. Each step adds its output at position idx via one-hot.
    _MAX_CHUNKS = 64  # covers seq <= 4096

    def _take(x, i):
        # x: [B, H, C, ...] -> [B, H, 1, ...]
        return torch.index_select(x, 2, i.reshape(1))

    def _cond(state, idx, acc):
        return idx < num_chunks

    def _body(state, idx, acc):
        # v_new = nv - kc @ S
        v_new = _take(new_values, idx) - torch.einsum(
            "bhock,bhkv->bhocv", _take(k_cumdecay, idx), state
        )
        # out = q @ S + ica @ v_new
        out_i = torch.einsum(
            "bhock,bhkv->bhocv", _take(query, idx), state
        ) + torch.einsum(
            "bhocd,bhodv->bhocv", _take(intra_chunk_attn, idx), v_new
        )
        # S' = S * d + k^T @ v_new  (d via einsum; no views allowed)
        d_bh = _take(chunk_decay, idx).sum(dim=(2, 3, 4))  # [B,H]
        next_state = torch.einsum(
            "bhkv,bh->bhkv", state, d_bh
        ) + torch.einsum("bhock,bhocv->bhkv", _take(key, idx), v_new)
        # write out_i into acc at position idx (static 64-wide one-hot)
        oh = torch.nn.functional.one_hot(idx, _MAX_CHUNKS).to(out_i.dtype)
        acc = acc + oh.view(1, 1, -1, 1, 1) * out_i
        return (next_state, idx + 1, acc)

    acc0 = torch.zeros(
        batch_size,
        num_v_heads,
        _MAX_CHUNKS,
        chunk_size,
        v_head_dim,
        dtype=new_values.dtype,
        device=new_values.device,
    )
    last_recurrent_state, _, core_attn_out = while_loop(
        _cond,
        _body,
        (last_recurrent_state, torch.tensor(0, device=new_values.device), acc0),
    )
    # core_attn_out is [B,H,64,chunk,V] (zeros beyond num_chunks); the
    # reshape + sequence_length slice below discards the excess, exactly as
    # the original code discards intra-chunk padding.

    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )
    return core_attn_out, last_recurrent_state
