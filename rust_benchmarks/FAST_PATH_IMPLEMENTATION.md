# Implementing the Variable Fast Path in `process_mul`

## The Problem in Concrete Terms

For `minimize ||Ax - b||²` with `A` being `m×n` (e.g. 5000×1000), CVXPY builds the
LinOp tree:

```
Mul(
  data = DenseConst(A),   // the 5000×1000 matrix
  args = [Variable(x)]    // the 1000-dimensional variable
)
```

`process_mul` in `arithmetic.rs` currently does this:

**Step 1** — process the arg (Variable):
`process_variable` in `leaf.rs` produces an identity tensor of `n` entries:
```
rows   = [0,   1,   2,   ..., n-1]
cols   = [c,   c+1, c+2, ..., c+n-1]   // c = col_offset of x in the variable block
data   = [1.0, 1.0, 1.0, ..., 1.0]
params = [p,   p,   p,   ..., p]        // p = const_param (non-parametric slice)
```

**Step 2** — call `multiply_dense_block_diagonal_colmajor(A, identity_tensor)`:
For each of the `n` entries in the identity tensor (rhs_val = 1.0):
- `rhs_row = i` → `col_in_block = i`, `col_start = i * a_rows`
- Emits the entire column `i` of A: `m` entries multiplied by `1.0`

Total work: `n * m` scalar multiplications (`val * 1.0` — literally multiplying by 1).
Total entries output: `n * m` = 5,000,000.

**This is the waste.** Multiplying by 1.0 `n` times, once per column of A.

---

## What the Fast Path Does Instead

The Jacobian of `f(x) = Ax` with respect to `x` is just `A`. No multiplication needed.

The fast path skips `process_variable` and `multiply_block_diagonal` entirely and
directly maps A's data into the output tensor:

For each nonzero `A[row, col]` in A:
- output `row` = `row` (same matrix row)
- output `col` = `col_offset_of_x + col` (map A's column index to variable column)
- output `value` = `A[row, col]`
- output `param_offset` = `const_param`

That's it. No multiplication. Just an index remap.

For dense A: iterate over the flat column-major array. For sparse A: iterate over CSC
nonzeros. Both are pure data copies with index arithmetic.

---

## How to Detect the Fast Path

The condition: **the arg of Mul is a plain Variable** (possibly wrapped in Reshape or
single-arg Sum, which are both no-ops in COO format).

Add a helper function to unwrap no-op wrappers and check:

```rust
/// Returns Some(var_id) if this LinOp is a plain variable (possibly wrapped
/// in Reshape or single-arg Sum, which are no-ops in COO format).
fn as_plain_variable(lin_op: &LinOp) -> Option<i64> {
    match lin_op.op_type {
        OpType::Variable => {
            if let LinOpData::Int(id) = lin_op.data {
                Some(id)
            } else {
                None
            }
        }
        // Reshape is always a no-op in COO format — unwrap it
        OpType::Reshape | OpType::Sum if lin_op.args.len() == 1 => {
            as_plain_variable(&lin_op.args[0])
        }
        _ => None,
    }
}
```

---

## Where to Insert the Check

In `process_mul` in `arithmetic.rs`, after extracting the `lhs_linop` and before
processing the rhs:

```rust
pub fn process_mul(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    let lhs_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Mul operation must have LinOp data"),
    };

    // ── NEW FAST PATH ──────────────────────────────────────────────────────────
    // If lhs is a non-parametric constant AND rhs is a plain variable,
    // the result is just A with column indices remapped to the variable block.
    // This avoids building the identity tensor and doing A @ I multiplication.
    if !is_parametric(lhs_linop) {
        if let Some(var_id) = as_plain_variable(&lin_op.args[0]) {
            let lhs_data = get_constant_matrix_data(lhs_linop, Some(ctx));
            return mul_const_by_variable(lhs_data, var_id, lin_op, ctx);
        }
    }
    // ──────────────────────────────────────────────────────────────────────────

    // ... rest of the existing function unchanged
}
```

---

## The Fast Path Function

```rust
/// Fast path for Mul(ConstantMatrix, Variable).
///
/// Instead of building an identity tensor for the variable and doing
/// block-diagonal matrix multiplication, we directly remap A's column
/// indices to the variable's column offset in the output tensor.
///
/// For dense A (m×n): maps A[row, col] → (row, var_col + col, A[row,col])
/// For sparse A (CSC): same mapping over nonzero entries
/// Cost: O(nnz(A)) — no multiplications, just index arithmetic + copy
fn mul_const_by_variable(
    lhs: ConstantMatrix,
    var_id: i64,
    lin_op: &LinOp,
    ctx: &ProcessingContext,
) -> SparseTensor {
    let output_rows = lin_op.size();
    let var_col_offset = ctx.var_col(var_id);
    let param_offset = ctx.const_param();

    match lhs {
        ConstantMatrix::Scalar(s) => {
            // Scalar * variable: scale the identity
            // This is the same as the existing scalar path, already fast
            let n = lin_op.args[0].size(); // size of variable after unwrapping
            let mut result = SparseTensor::with_capacity(
                (output_rows, ctx.var_length as usize + 1), n
            );
            for i in 0..n {
                if s != 0.0 {
                    result.push(s, i as i64, var_col_offset + i as i64, param_offset);
                }
            }
            result
        }

        ConstantMatrix::DenseColMajor { data, rows: a_rows, cols: a_cols } => {
            // Dense column-major A: data is laid out as [col0_row0, col0_row1, ..., col1_row0, ...]
            // For each (col, row) pair, the output entry is:
            //   output_row = col * a_rows + row   ... wait, no.
            //
            // A is m×n (a_rows × a_cols). Output row = matrix row (0..a_rows-1).
            // Variable column = var_col_offset + matrix_col.
            //
            // In column-major storage: data[col * a_rows + row] = A[row, col]
            let nnz_estimate = data.iter().filter(|&&v| v != 0.0).count();
            let mut result = SparseTensor::with_capacity(
                (output_rows, ctx.var_length as usize + 1), nnz_estimate
            );

            for col in 0..a_cols {
                let out_col = var_col_offset + col as i64;
                let col_start = col * a_rows;
                for row in 0..a_rows {
                    let val = data[col_start + row];
                    if val != 0.0 {
                        result.push(val, row as i64, out_col, param_offset);
                    }
                }
            }
            result
        }

        ConstantMatrix::DenseRowMajor { data, rows: a_rows, cols: a_cols } => {
            // Row-major: data[row * a_cols + col] = A[row, col]
            // This is used for 1D arrays treated as row vectors (a_rows == 1)
            let nnz_estimate = data.iter().filter(|&&v| v != 0.0).count();
            let mut result = SparseTensor::with_capacity(
                (output_rows, ctx.var_length as usize + 1), nnz_estimate
            );

            for row in 0..a_rows {
                for col in 0..a_cols {
                    let val = data[row * a_cols + col];
                    if val != 0.0 {
                        let out_col = var_col_offset + col as i64;
                        result.push(val, row as i64, out_col, param_offset);
                    }
                }
            }
            result
        }

        ConstantMatrix::Sparse { values, row_indices, col_indptr, rows: _, cols: a_cols } => {
            // CSC sparse: iterate over columns, then nonzeros within each column
            let mut result = SparseTensor::with_capacity(
                (output_rows, ctx.var_length as usize + 1), values.len()
            );

            for col in 0..a_cols {
                let out_col = var_col_offset + col as i64;
                let start = col_indptr[col] as usize;
                let end = col_indptr[col + 1] as usize;
                for idx in start..end {
                    let row = row_indices[idx];
                    let val = values[idx];
                    if val != 0.0 {
                        result.push(val, row, out_col, param_offset);
                    }
                }
            }
            result
        }
    }
}
```

---

## Why This Is Correct

The `multiply_dense_block_diagonal_colmajor` function computes `kron(I_k, A) @ rhs`.

When `rhs` is the identity tensor for a single variable (i.e. `k=1`, `rhs` is I_n):

```
kron(I_1, A) @ I_n  =  A @ I_n  =  A
```

The output has:
- Row `r` (0..m-1) → comes from row `r` of A
- Column `var_col_offset + c` → comes from column `c` of A

The fast path constructs exactly this output, entry by entry, without any multiplication.

The existing path arrives at the same result via:
1. Building I_n as a tensor (n entries, all `1.0`)
2. For each of the n entries in I_n, emitting one column of A scaled by `1.0`
3. `n * m` operations, `val * 1.0` every time

Same output, unnecessary work eliminated.

---

## The `k > 1` Block Case

The block structure (`kron(I_k, A)`) appears when a variable has been reshaped such that
the same matrix A applies to multiple blocks. In this case the rhs tensor has more than
n entries — there are `k*n` entries (one identity block per repeat).

The simple fast path above (checking `as_plain_variable`) only fires for a direct
`Variable` arg (k=1). This is the overwhelmingly common case and covers your benchmark.

The block case (`k > 1`) would need the variable to be wrapped in a Reshape with a
different shape. You can extend the fast path later to handle this by reading how many
blocks the output requires from `lin_op.size() / a_rows`.

---

## What to Test

After implementing, run:

```bash
# Quick correctness check
python -m pytest cvxpy/tests/test_rust_backend.py -x -v

# Benchmark to confirm improvement
python rust_benchmarks/rustybench.py
```

The test suite compares Rust output against SciPy for all 22 LinOp types. If the fast
path produces wrong column indices or wrong values, `test_mul` will catch it.

For the benchmark you should expect the RUST time to drop significantly for the
least-squares case (n=1000, m=5000) — the 5M scalar multiplications become a single
O(nnz) copy loop.

---

## Summary

| | Current path | Fast path |
|---|---|---|
| Operations | Build I_n tensor, then `A @ I_n` | Direct index remap of A's data |
| Multiplications | `n * m` (5M for your benchmark) | 0 |
| Memory allocations | Identity tensor + output tensor | Output tensor only |
| Correctness condition | Always correct | Fires only when arg is plain Variable |
| Code change | — | ~50 lines in `arithmetic.rs` |

The change is localized entirely to `arithmetic.rs`. No other files need to change.
