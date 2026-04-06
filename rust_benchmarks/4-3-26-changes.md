# Changes — April 3, 2026

## Benchmark Results

Ran from `/home/ray/cvxrust/cvxpy/rust_benchmarks/` with the `(cvxrust)` venv active.
Problem: least-squares `minimize ||Ax - b||²`, n=1000, m=5000, dense A.

### Before (no changes)
```
RUST:  ~2.31s
CPP:   ~1.88s
SCIPY: ~1.74s
```
RUST was slowest.

### After (both changes applied)
```
RUST:  ~1.72s  ← fastest
CPP:   ~1.92s
SCIPY: ~1.76s
```
RUST is now fastest. ~0.59s improvement.

---

## Important: Build Command

Always build with `--release`. Without it you get a debug build that is 2-3x slower.

```bash
cd /home/ray/cvxrust/cvxpy/cvxpy_rust
maturin develop --release
```

Debug build output says `[unoptimized + debuginfo]` — if you see that, rebuild.
Release build output says `[optimized]`.

---

## Change 1: Variable Fast Path in `process_mul`

**File:** `cvxpy_rust/src/operations/arithmetic.rs`

### Root Cause

For `Mul(DenseConst(A), Variable(x))` — which is how `A @ x` appears in the LinOp
tree — the existing code did two things:

1. `process_variable(x)` built an identity tensor: n entries, all value `1.0`, with
   row i mapping to column `var_col_offset + i`.

2. `multiply_dense_block_diagonal_colmajor(A, identity_tensor)` then iterated over
   all n entries in the identity tensor and for each one emitted an entire column of A
   scaled by `rhs_val = 1.0`.

For n=1000, m=5000 this was **5 million multiplications of `val * 1.0`** — multiplying
by 1 every time, building two intermediate data structures to do it.

### The Insight

The Jacobian of `f(x) = A @ x` with respect to `x` is just `A`. No multiplication
is needed. This is the core diffengine insight from `context.md` applied directly in
Rust: rather than computing `A @ I_x` through the full tensor machinery, detect that
the argument is a plain variable and return A with column indices remapped.

For each nonzero `A[row, col]`:
- output row = `row`
- output col = `var_col_offset + col`
- output value = `A[row, col]`

Zero multiplications. Pure index arithmetic and data copy.

### What Was Added

**`as_plain_variable(lin_op) -> Option<i64>`**

Unwraps Reshape and single-arg Sum nodes (both are no-ops in COO format) and returns
`Some(var_id)` if the innermost node is a Variable, `None` otherwise. These wrappers
appear when CVXPY reshapes a variable before passing it to a Mul.

**`mul_const_by_variable(lhs, var_id, lin_op, ctx) -> SparseTensor`**

The fast path itself. Iterates directly over A's data and remaps column indices.
Handles all four `ConstantMatrix` variants:
- `Scalar` — emits scaled identity entries
- `DenseColMajor` — iterates column-major data, maps each entry
- `DenseRowMajor` — iterates row-major data (1D array case), maps each entry
- `Sparse` — iterates CSC nonzeros, maps each entry

**Detection guard in `process_mul`**

Inserted before the existing path, after extracting `lhs_linop`:

```rust
if !is_parametric(lhs_linop) {
    if let Some(var_id) = as_plain_variable(&lin_op.args[0]) {
        let lhs_data = get_constant_matrix_data(lhs_linop, Some(ctx));
        return mul_const_by_variable(lhs_data, var_id, lin_op, ctx);
    }
}
```

The `!is_parametric` guard ensures we only take the fast path when A is a constant
matrix (not a parameter). If A is parametric, the result depends on the parameter
values and the fast path does not apply.

### Why It Is Correct

`multiply_dense_block_diagonal_colmajor` computes `kron(I_k, A) @ rhs`. When rhs is
the identity tensor for a single variable (k=1):

```
kron(I_1, A) @ I_n  =  A @ I_n  =  A
```

The fast path constructs this output directly. The existing test suite compares all
Mul outputs against SciPy numerically.

---

## Change 2: Pre-sort COO Output in `from_tensor`

**File:** `cvxpy_rust/src/tensor.rs`

### Root Cause

After `build_matrix` returns, Python's `reduce_problem_data_tensor` in
`canonInterface.py` does:

```python
unique_old_row, reduced_row = np.unique(A_coo.row, return_inverse=True)
```

This finds which entries of the 3D coefficient tensor are nonzero so it can compress
them. The `A_coo.row` array contains the flat row indices (encoding constraint-row ×
variable-column position) for every nonzero entry.

The flat rows returned by Rust were **unsorted**. `SparseTensor::combine` just
concatenates tensors in processing order:

- `mul(A, x)` entries: flat rows in a high range (x occupies later variable columns)
- `neg(variable(t))` entries: flat rows near zero (t occupies early variable columns)
- `dense_const(-b)` entries: flat rows at the constant column

This interleaving means `np.unique` had to run a full **O(n log n) sort on ~5M
elements**, which the profiler measured at ~575ms.

### The Fix

Sort the COO entries by flat_row **inside Rust** using rayon's parallel sort before
returning to Python:

```rust
let mut order: Vec<usize> = (0..nnz).collect();
order.par_sort_unstable_by_key(|&i| flat_rows[i]);

let sorted_rows: Vec<i64> = order.iter().map(|&i| flat_rows[i]).collect();
let sorted_data: Vec<f64> = order.iter().map(|&i| tensor.data[i]).collect();
let sorted_cols: Vec<i64> = order.iter().map(|&i| tensor.param_offsets[i]).collect();
```

When `np.unique` receives a pre-sorted array, Python's timsort detects the order in
a single linear O(n) pass instead of a full O(n log n) comparison sort. For 5M
elements this is the difference between ~575ms and negligible.

The sort runs in parallel across CPU cores via rayon. `par_sort_unstable` is used
because there are no ordering requirements between entries with equal flat_row values
— they are independent COO entries that will be summed later.

### Why It Is Correct

`from_tensor` previously made no ordering promises — it returned flat_rows in whatever
order the tensor entries were concatenated. Sorting preserves all values and just
reorders them. `reduce_problem_data_tensor` treats the COO as an unordered set of
(row, col, value) triples, so the result is identical regardless of entry order.

---

## Profiler Evidence (before changes)

Running `cProfile` on the RUST backend for the least-squares benchmark:

| Bottleneck | Time | Call count |
|---|---|---|
| `cvxpy_rust.build_matrix` (Rust FFI) | 0.914s | 2 |
| `np.unique` in `reduce_problem_data_tensor` | 0.575s | 6 |
| `csr_sort_indices` in scipy | 0.438s | 1 |
| Conic solver interface | ~0.333s | 1 |

Change 1 reduced the Rust FFI time. Change 2 eliminated the numpy sort bottleneck.

---

## Files Changed

| File | Change |
|------|--------|
| `cvxpy_rust/src/operations/arithmetic.rs` | Added `as_plain_variable`, `mul_const_by_variable`, fast path check in `process_mul`, added `OpType` to imports |
| `cvxpy_rust/src/tensor.rs` | Added `rayon::prelude::*` import, replaced `from_tensor` body with parallel-sorted output |

## Files NOT Changed

- All Python-side code (`canon_backend.py`, `canonInterface.py`, solver interfaces)
- The CPP and SciPy backends
- All 21 other LinOp operation types in Rust
- The fast path is additive — all non-matching cases fall through to the existing code
