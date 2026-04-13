# Benchmarking the CVXPY Rust Canonicalization Backend

Comprehensive guide covering research findings, benchmark suite implementation, cross-platform analysis, and usage instructions.

---

## Table of Contents

1. [What We're Benchmarking](#what-were-benchmarking)
2. [Why Results Differ Across Machines](#why-results-differ-across-machines)
3. [How CVXPY Profiles Its Backends](#how-cvxpy-profiles-its-backends)
4. [The Benchmark Suite (`benchmark_suite.py`)](#the-benchmark-suite)
5. [Implementation Details](#implementation-details)
6. [Problem Suite](#problem-suite)
7. [Scaling Analysis](#scaling-analysis)
8. [Interpreting Results](#interpreting-results)
9. [Usage Reference](#usage-reference)
10. [Results from Linux Desktop (April 2026)](#results-from-linux-desktop)
11. [Legacy Benchmarks](#legacy-benchmarks)

---

## What We're Benchmarking

The CVXPY canonicalization pipeline transforms a user-written optimization problem into a solver-ready sparse matrix form. The pipeline has these stages:

```
User Problem (objective + constraints)
    |
    v
[1] Python reductions (DCP checking, cone transformations)   <-- shared across all backends
    |
    v
[2] LinOp tree generation (expression DAG)                   <-- shared across all backends
    |
    v
[3] backend.build_matrix(linOps)                             <-- THIS is what differs
    |   RUST:  Python->Rust FFI -> rayon parallel tree walk -> sparse tensor ops -> back to Python
    |   SCIPY: Python tree traversal -> numpy/scipy tensor ops
    |   CPP:   Python->C++ via SWIG -> Eigen sparse ops -> back to Python
    |
    v
[4] Problem data assembly (post-processing)                  <-- shared across all backends
    |
    v
Solver (CLARABEL, SCS, OSQP, etc.)                          <-- NOT benchmarked
```

**We only benchmark canonicalization, not solving.** The benchmark calls `problem.get_problem_data(solver, canon_backend=backend)` which stops before invoking the solver.

The benchmark suite measures at two isolation layers:
- **End-to-end**: Stages 1-4 (includes shared Python overhead)
- **build_matrix only**: Stage 3 only (isolates the actual backend difference)

The `build_matrix` layer is the primary metric because it removes ~50% of shared overhead that is identical across backends.

---

## Why Results Differ Across Machines

### The Core Issue: BLAS Libraries

The single biggest factor in cross-platform variance is which **BLAS library** numpy/scipy link against:

| Platform | Typical BLAS | Characteristics |
|----------|-------------|-----------------|
| **macOS** | Apple Accelerate | Hardware-optimized for Apple Silicon (AMX coprocessor). Very fast for dense operations. Ships with the OS. |
| **Linux (default)** | OpenBLAS | Open-source, good general performance. Relies on NEON/SSE/AVX instructions. |
| **Linux (Intel)** | MKL | Intel's proprietary library. 2x faster than OpenBLAS on Intel CPUs for some operations. |

**What this means for benchmarks:**

- The **SciPy backend** delegates dense matrix operations to BLAS. Its performance is really "BLAS performance" for dense-heavy problems. On macOS with Accelerate, SciPy can be significantly faster than on Linux with OpenBLAS.
- The **C++ backend** (Eigen) has its own BLAS-like optimized kernels. Its relative performance vs SciPy depends on which BLAS SciPy is using.
- The **Rust backend** does its own sparse operations with no BLAS dependency. Its absolute performance should be more consistent across platforms.

This is why you see:
- **macOS**: SciPy slower than CPP (Accelerate helps CPP's Eigen more than SciPy's sparse path)
- **Linux**: SciPy faster than CPP (OpenBLAS is well-optimized for SciPy's access patterns)

### Other Sources of Variance

1. **Thread scheduling**: Rust uses rayon for parallelism, SciPy may use BLAS threads, CPP uses Eigen's threading. Different thread counts and scheduling policies affect results.
2. **CPU architecture**: AMD vs Intel, cache sizes, SIMD instruction sets (AVX-512 vs AVX2 vs NEON)
3. **JIT/cache warming**: First invocation is slower due to shared library loading, page faults, instruction cache misses
4. **Python version and GC behavior**: Different Python versions have different overhead characteristics

### The Solution: Speedup Ratios

Absolute times are not comparable across machines. **Speedup ratios** (e.g., SCIPY/RUST) are much more stable because:
- Shared Python overhead cancels in the ratio
- The `build_matrix` layer isolates the actual algorithm difference
- BLAS differences primarily affect one side (SCIPY) not the other (RUST)

The benchmark suite uses geometric mean of speedup ratios as the primary cross-machine metric. Geometric mean is standard in benchmarking (SPEC CPU, etc.) because it avoids ratio bias: a 4x speedup and 0.25x regression average to 2.125x arithmetically but are actually neutral (geometric mean = 1.0).

---

## How CVXPY Profiles Its Backends

### Upstream CVXPY Infrastructure

CVXPY's CI/CD runs benchmarks via GitHub Actions:
- **`benchmarks.yml`**: Runs on push to master, triggers upstream benchmark workflow
- **`pr_benchmarks.yml`**: Runs on pull requests
- **`pr_benchmark_comment.yml`**: Posts benchmark results as PR comments
- **`test_backends.yml`**: Correctness verification across all backends using `CVXPY_CANON_BACKEND` env var

### Backend Correctness Verification

All backends inherit from the `CanonBackend` abstract class and implement `build_matrix(linOps) -> csc_array`. Correctness is verified by:
- Running the full pytest suite with each backend (`CVXPY_DEFAULT_CANON_BACKEND=RUST pytest`)
- `test_python_backends.py`: Tests 24 LinOp operation types numerically against SciPy baseline
- `test_rust_backend.py`: Rust-specific tests

### Historical Profiling Approaches

Previous profiling in this project used:
- **cProfile**: Function-level bottleneck identification (see `profile_rust_backend.py`)
- **Subprocess isolation**: Cold-start measurements (see `quick_benchmark.py`)
- **In-process iteration**: Warmup + timed iterations (see `benchmark_rust_backend.py`)

The new `benchmark_suite.py` consolidates all these approaches into a single tool.

---

## The Benchmark Suite

### Architecture

`benchmark_suite.py` is a single self-contained script (~600 lines) with these components:

1. **Environment fingerprinting** -- captures BLAS, CPU, versions, thread settings
2. **Measurement engine** -- statistical timing with GC control, warmup, confidence intervals
3. **build_matrix isolation** -- monkey-patches `canonInterface.get_problem_matrix` to capture arguments
4. **Problem suite** -- 40+ problem configurations across 5 categories
5. **Scaling analysis** -- sweeps constraint count, variable size, and matrix density
6. **Output** -- human-readable tables + structured JSON for cross-machine comparison
7. **Cold-start mode** -- subprocess isolation for first-invocation latency
8. **Compare mode** -- aligns two JSON result files and highlights disagreements

### Key Design Decisions

**Why monkey-patching for build_matrix isolation:**

The `canonInterface.get_problem_matrix` function (line 258 of `canonInterface.py`) is the single point where the backend is selected and `build_matrix` is called. By intercepting this function, we capture the exact `linOps`, `id_to_col`, `param_to_size`, etc. that the real pipeline produces, then replay them against each backend in isolation. This avoids reimplementing the reduction chain and guarantees we benchmark the exact same workload.

**Why GC is disabled during measurement:**

Python's garbage collector can trigger at any time, adding milliseconds of jitter. The suite calls `gc.collect()` before each measurement to clear pending finalizations, then disables GC during the timed region.

**Why fresh Problem instances per iteration:**

`Problem.get_problem_data()` caches the `param_prog` on the Problem instance. A second call on the same Problem bypasses canonicalization entirely. Every iteration must use a fresh Problem from the factory.

**Why `perf_counter_ns` instead of `perf_counter`:**

For sub-millisecond measurements, `perf_counter` (float64) has precision loss. `perf_counter_ns` returns integer nanoseconds with no precision loss, then we divide by 1e6 for milliseconds.

**Why thread control defaults:**

By default, the suite sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` to prevent SciPy from using BLAS multi-threading. This ensures we measure algorithm efficiency, not "BLAS can use 12 threads." The Rust backend's rayon threads are left at their default (all cores) unless `--single-thread` is passed.

---

## Implementation Details

### Environment Fingerprinting

Captured at the start of every run:

```
Machine: Linux Linux-6.17.0-20-generic-x86_64-with-glibc2.39 / x86_64 / 12 cores
CPU: AMD Ryzen 5 7600X 6-Core Processor
BLAS: scipy-openblas 0.3.31.dev
Python 3.12.3 / NumPy 2.4.3 / SciPy 1.17.1 / CVXPY 1.8.0.dev0
Rust backend: available
Threads: RAYON=auto, OMP=1
Git: 866f39957
```

The **BLAS line** is the most important piece of metadata. When comparing results from two machines, check this first -- it explains most performance differences for dense-heavy problems.

BLAS detection uses `np.show_config(mode="dicts")` (numpy >= 2.0) which returns a structured dict with the BLAS library name and version. Falls back to parsing `np.__config__.blas_opt_info` on older numpy.

### Measurement Engine

```
TimingResult:
  - raw_times_ms: [list of timed measurements]
  - warmup_times_ms: [list of warmup measurements, not used in statistics]
  - mean_ms, median_ms, std_ms, min_ms, max_ms
  - ci_95: 95% confidence interval via t-distribution
  - cv: coefficient of variation (std/mean), flagged if > 10%
```

**Adaptive iteration counts** based on estimated problem time:

| Estimated time | Warmup | Iterations (default) | Iterations (thorough) |
|---------------|--------|---------------------|----------------------|
| < 10ms        | 5      | 30                  | 60                   |
| < 100ms       | 3      | 15                  | 30                   |
| < 1000ms      | 2      | 7                   | 14                   |
| >= 1s          | 1      | 5                   | 10                   |

Quick mode: always (1, 5) regardless of problem size.

### build_matrix Isolation

The isolation works by monkey-patching `canonInterface.get_problem_matrix`:

```python
# 1. Replace get_problem_matrix with a capturing version
canonInterface.get_problem_matrix = capturing_fn

# 2. Run the full reduction chain once (with SCIPY to complete it)
prob.get_problem_data(solver, canon_backend="SCIPY")

# 3. Restore original function
canonInterface.get_problem_matrix = original_fn

# 4. For each backend, replay the captured arguments:
backend = CanonBackend.get_backend(name, dict(id_to_col), ...)
backend.build_matrix(captured_linOps)
```

**Critical: `id_to_col` mutation.** Both `PythonCanonBackend.build_matrix` (line 196 of `canon_backend.py`) and `RustCanonBackend.build_matrix` (line 698) insert `-1 -> var_length` into `id_to_col` at the start and remove it at the end. Each `time_build_matrix` call passes a fresh `dict(id_to_col)` copy to avoid mutation leaking between iterations.

**What the build_matrix layer measures (Rust backend):**
1. Python -> Rust FFI: LinOp tree extraction (WITH GIL held)
2. Rust `build_matrix_internal`: tree traversal + sparse tensor ops (GIL released)
3. Rust -> Python: result conversion to numpy arrays (WITH GIL held)

There is no way to separately time step 2 from Python without modifying the Rust code. The FFI overhead (steps 1 and 3) is typically < 1ms for medium problems.

---

## Problem Suite

### Category A: Arithmetic-Heavy (Mul/Rmul dominate)

These problems exercise dense and sparse matrix multiplication -- the core of most canonicalization workloads.

| Problem | Description | Sizes | Key LinOps |
|---------|-------------|-------|-----------|
| `dense_matmul` | min \|\|Ax-b\|\|^2, A dense | n=50,200,500,1000 | dense_const, mul, sum_entries |
| `sparse_matmul` | min \|\|Ax-b\|\|^2, A 5% sparse | n=100,500,2000 | sparse_const, mul, sum_entries |
| `dense_qp` | min x'Qx+c'x, Ax<=b | n=50,200,500 | mul, rmul, quad_form |

**Expected behavior**: Rust should be faster for small/medium sizes. For very large dense problems (n=1000), SciPy's BLAS advantage may show -- this is the LASSO slowdown case documented in `RUST_BACKEND_PERFORMANCE_ANALYSIS.md`.

### Category B: Constraint-Heavy (tests rayon parallelism)

These problems have many independent linear constraints, which is where Rust's rayon parallelism provides the biggest speedup.

| Problem | Description | Sizes | Key LinOps |
|---------|-------------|-------|-----------|
| `many_constraints` | many a'x <= b | m=10,50,100,500,1000,5000 | mul, constraint stacking |
| `box_constraints` | l <= x <= u | n=50,200,1000 | index, variable |

**Expected behavior**: Speedups of 10-40x for many constraints. The Rust backend's rayon parallelism threshold is at `PARALLEL_MIN_CONSTRAINTS=4` AND `PARALLEL_MIN_WORK=500` estimated non-zeros (see `matrix_builder.rs:14-18`).

### Category C: Structural (Index/Reshape/Stack)

These problems exercise structural operations that rearrange tensor entries without arithmetic.

| Problem | Description | Sizes | Key LinOps |
|---------|-------------|-------|-----------|
| `matrix_indexing` | min \|\|X[i:j,k:l]\|\|_F^2 | n=20,50,100 | index, reshape |
| `hstack` | many small exprs stacked | width=10,50,200 | hstack, mul |
| `portfolio` | max mu'w - gamma*w'Sigma*w | n=50,200,500 | mul, sum, quad_form |

**Expected behavior**: Moderate speedups (2-5x). The `hstack` problems should show larger speedups because they involve many small operations that SciPy handles with Python-level loops.

### Category D: Specialized

Problems that exercise specific LinOp types.

| Problem | Description | Sizes | Key LinOps |
|---------|-------------|-------|-----------|
| `lasso` | \|\|Ax-b\|\|^2 + lambda\|\|x\|\|_1 | n=50,200,500,1000 | sum_squares, norm1, mul |
| `svm` | SVM with slack variables | m=100,500 | mul, mul_elem |
| `convolution` | signal deconvolution | signal=100,500 | conv, sum_squares |

**Expected behavior**: LASSO (n=200+) is the known case where SciPy can win due to BLAS advantage for large dense matrix operations. SVM and convolution should favor Rust.

### Category E: Expression Depth

Tests how performance scales with expression tree depth.

| Problem | Description | Sizes | Key LinOps |
|---------|-------------|-------|-----------|
| `nested_affine` | A1@A2@...@Ad@x, each Ai is n x n | depth=3,5,10,20 | repeated mul |

**Expected behavior**: Moderate speedups. Deep trees test the recursive processing overhead.

---

## Scaling Analysis

Three sweeps run at the `build_matrix` layer only, with all other variables held constant.

### Constraint Count Scaling

- **Fixed**: n_vars=50
- **Swept**: n_constraints in [4, 10, 50, 100, 500, 1000, 2000, 5000]
- **Purpose**: Tests rayon parallelism. Below 4 constraints or 500 estimated nnz, Rust processes sequentially. Above the threshold, rayon parallelism kicks in and speedups increase dramatically.

### Variable Size Scaling

- **Fixed**: n_constraints=10
- **Swept**: n_vars in [10, 50, 100, 500, 1000, 2000, 5000]
- **Purpose**: Tests how matrix dimensions affect performance. For the Rust backend, matrix size affects sparse tensor allocation and iteration. For SciPy, larger matrices benefit more from BLAS.

### Matrix Density Scaling

- **Fixed**: n=500, m=200
- **Swept**: density in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
- **Purpose**: Tests the sparse-vs-dense tradeoff. At low density, both backends work with sparse representations. At high density, SciPy may benefit from BLAS dense routines while Rust still uses sparse representations.

---

## Interpreting Results

### Reading the Console Output

```
  Problem                                   RUST      SCIPY   SCIPY/RUST
  many_constraints (m=500)                2.91ms    108.34ms    37.27x
  lasso (n=200)                           6.38ms      4.65ms     0.73x
```

- Times are mean across all iterations for the selected layer (default: build_matrix)
- `!` after a time means CV > 10% (high variance, less reliable)
- Speedup > 1.0x means first backend (RUST) is faster
- Speedup < 1.0x means second backend (SCIPY) is faster

### Summary Statistics

```
Layer: build_matrix
  SCIPY/RUST: geomean 3.56x, arith mean 7.79x, range [0.73x, 42.41x], RUST wins 26/27
```

- **geomean**: Geometric mean of speedup ratios -- the primary metric. Less sensitive to outliers than arithmetic mean.
- **arith mean**: Arithmetic mean -- included for reference but can be skewed by extreme speedups.
- **range**: Worst and best case speedups.
- **wins**: Number of problems where the first backend is faster.

### High Variance Warnings

```
High variance (CV>10%): dense_matmul (n=50)/RUST (CV=13%), ...
```

Results with CV > 10% should be treated with caution. This typically happens for sub-millisecond measurements where timing precision approaches the noise floor. Use `--thorough` mode or increase problem sizes for more reliable results.

### Comparing Across Machines

Use `--json` to save results, then `compare` to analyze:

```bash
# On machine A
python3 benchmark_suite.py --json results_mac.json

# On machine B  
python3 benchmark_suite.py --json results_linux.json

# Compare
python3 benchmark_suite.py compare results_mac.json results_linux.json
```

The compare output shows speedup ratios from both machines side by side, with an "Agree?" column indicating whether both machines agree on which backend is faster. Disagreements are the interesting cases -- they indicate problems where the BLAS difference flips the winner.

---

## Usage Reference

### Basic Usage

```bash
# Default: build_matrix layer, RUST vs SCIPY, all sizes, with scaling
python3 benchmark_suite.py

# Quick run (fewer iterations, good for development)
python3 benchmark_suite.py --quick

# Thorough run (more iterations, for final results)
python3 benchmark_suite.py --thorough

# Only small and medium problems
python3 benchmark_suite.py --sizes small,medium

# Also include end-to-end layer
python3 benchmark_suite.py --layers all
```

### Backend Selection

```bash
# Compare RUST against SCIPY (default)
python3 benchmark_suite.py --backends RUST SCIPY

# Include CPP backend
python3 benchmark_suite.py --backends RUST SCIPY CPP

# Only SCIPY (for profiling SciPy itself)
python3 benchmark_suite.py --backends SCIPY
```

### Thread Control

```bash
# Default: BLAS threading disabled (OMP=1), rayon at full cores
python3 benchmark_suite.py

# Pure algorithmic comparison (all threads to 1)
python3 benchmark_suite.py --single-thread
```

The default is recommended for understanding "how fast is the backend's algorithm." The `--single-thread` mode is useful for understanding "how fast without any parallelism" -- it reveals how much of Rust's speedup comes from rayon vs algorithmic efficiency.

### Output Options

```bash
# Save JSON for cross-machine comparison
python3 benchmark_suite.py --json results.json

# Skip scaling analysis (faster)
python3 benchmark_suite.py --no-scaling

# Also run cold-start (subprocess) benchmarks
python3 benchmark_suite.py --cold-start

# Compare two machines
python3 benchmark_suite.py compare machine_a.json machine_b.json
```

### Full CLI Reference

```
python3 benchmark_suite.py [options]

Options:
  --backends B [B ...]     Backends to benchmark (default: RUST SCIPY)
  --quick                  1 warmup, 5 iterations per problem
  --thorough               Double the default iteration counts
  --single-thread          Set all thread counts (BLAS + rayon) to 1
  --sizes SIZES            Filter: small,medium,large,all (default: all)
  --layers {all,e2e,bm}    Isolation layers to run (default: bm)
  --scaling / --no-scaling Include scaling analysis (default: yes)
  --json FILE              Write JSON results to file
  --seed SEED              Random seed (default: 42)
  --cold-start             Also run cold-start (subprocess) benchmarks

Subcommands:
  compare FILE1 FILE2      Compare two JSON result files from different machines
```

---

## Results from Linux Desktop (April 2026)

### Machine

```
CPU: AMD Ryzen 5 7600X 6-Core Processor (12 threads)
BLAS: scipy-openblas 0.3.31.dev
Python 3.12.3 / NumPy 2.4.3 / SciPy 1.17.1 / CVXPY 1.8.0.dev0
```

### Summary (build_matrix layer, quick mode)

```
Geometric mean speedup (RUST/SCIPY): 3.56x
RUST wins: 26/27 problems
Min speedup: 0.73x (lasso n=200)
Max speedup: 42.41x (hstack width=50)
```

### Results by Category

**Arithmetic-Heavy:**
| Problem | RUST (ms) | SCIPY (ms) | Speedup |
|---------|-----------|------------|---------|
| dense_matmul (n=50) | 0.09 | 0.19 | 2.05x |
| dense_matmul (n=200) | 0.15 | 0.33 | 2.22x |
| sparse_matmul (n=100) | 0.16 | 0.21 | 1.31x |
| sparse_matmul (n=500) | 0.08 | 0.16 | 2.00x |
| dense_qp (n=50) | 0.11 | 0.45 | 4.09x |
| dense_qp (n=200) | 0.13 | 0.46 | 3.60x |

**Constraint-Heavy (Rust dominates):**
| Problem | RUST (ms) | SCIPY (ms) | Speedup |
|---------|-----------|------------|---------|
| many_constraints (m=10) | 0.23 | 2.48 | 10.60x |
| many_constraints (m=50) | 0.46 | 11.14 | 24.23x |
| many_constraints (m=100) | 0.73 | 23.06 | 31.71x |
| many_constraints (m=500) | 2.91 | 108.34 | 37.27x |

**The One Loss (BLAS advantage):**
| Problem | RUST (ms) | SCIPY (ms) | Speedup |
|---------|-----------|------------|---------|
| lasso (n=200) | 6.38 | 4.65 | 0.73x |

### Constraint Count Scaling

```
 m=     4:  RUST  0.18ms, SCIPY   1.22ms  ( 6.71x)
 m=    10:  RUST  0.38ms, SCIPY   2.70ms  ( 7.20x)
 m=    50:  RUST  1.42ms, SCIPY  17.16ms  (12.07x)
 m=   100:  RUST  1.14ms, SCIPY  28.95ms  (25.40x)
 m=   500:  RUST  3.59ms, SCIPY 118.87ms  (33.14x)
 m=  1000:  RUST  5.99ms, SCIPY 233.44ms  (38.99x)
 m=  2000:  RUST 17.55ms, SCIPY 452.46ms  (25.78x)
 m=  5000:  RUST 34.37ms, SCIPY   1.11s   (32.20x)
```

The speedup increases dramatically with constraint count due to rayon parallelism. The slight dip at m=2000 vs m=1000 is likely due to memory allocation overhead at larger scales.

---

## Legacy Benchmarks

These files predate `benchmark_suite.py` and are kept for reference:

### `rustybench.py`
- **What**: Single LASSO problem (n=1000, m=5000), one measurement per backend
- **Limitation**: No statistics, no warmup, measures a single dense-heavy problem that is Rust's worst case
- **When to use**: Quick smoke test that backends work

### `quick_benchmark.py`
- **What**: 6 problems with cold-start subprocess methodology (10 samples each)
- **Limitation**: Measures end-to-end including Python import/startup overhead, slow to run
- **When to use**: When you specifically need cold-start (first-invocation) measurements. The `--cold-start` flag in `benchmark_suite.py` provides a similar capability.

### `benchmark_rust_backend.py`
- **What**: 22-problem comprehensive suite with warmup + iterations
- **Limitation**: In-process only, no build_matrix isolation, no environment fingerprinting, no JSON output
- **When to use**: Superseded by `benchmark_suite.py`

### `profile_rust_backend.py`
- **What**: cProfile profiling with constraint/variable/density scaling, operation breakdown
- **Limitation**: Uses cProfile which adds its own overhead, good for finding bottlenecks not for performance measurement
- **When to use**: When you need to identify which Python functions are slow (e.g., "is `np.unique` the bottleneck?"). Not suitable for accurate timing comparison.

---

## Key Files Reference

### Benchmark Files

| File | Purpose |
|------|---------|
| `rust_benchmarks/benchmark_suite.py` | Primary benchmark tool |
| `rust_benchmarks/rustybench.py` | Quick smoke test |
| `rust_benchmarks/quick_benchmark.py` | Cold-start subprocess benchmarks |
| `rust_benchmarks/profile_rust_backend.py` | cProfile bottleneck analysis |
| `rust_benchmarks/benchmark_rust_backend.py` | Legacy comprehensive benchmark |

### Rust Backend Source

| File | Purpose |
|------|---------|
| `cvxpy_rust/src/lib.rs` | PyO3 entry point, GIL release boundary |
| `cvxpy_rust/src/matrix_builder.rs` | Core algorithm, rayon parallelism thresholds |
| `cvxpy_rust/src/linop.rs` | LinOp extraction from Python (22 operation types) |
| `cvxpy_rust/src/tensor.rs` | SparseTensor COO representation, combine/flatten |
| `cvxpy_rust/src/operations/` | Per-operation implementations (leaf, arithmetic, structural, specialized) |

### CVXPY Backend Interface

| File | Purpose |
|------|---------|
| `cvxpy/lin_ops/canon_backend.py` | CanonBackend ABC, RustCanonBackend, SciPyCanonBackend |
| `cvxpy/cvxcore/python/canonInterface.py` | `get_problem_matrix()` -- dispatch to backends |
| `cvxpy/reductions/canonicalization.py` | Expression tree canonicalization |
| `cvxpy/problems/problem.py` | `Problem.get_problem_data()` entry point |

### Previous Analysis

| File | Purpose |
|------|---------|
| `rust_benchmarks/RUST_BACKEND_PERFORMANCE_ANALYSIS.md` | Historical performance analysis |
| `rust_benchmarks/4-3-26-changes.md` | Variable fast path and pre-sort optimizations |
| `rust_benchmarks/FAST_PATH_IMPLEMENTATION.md` | Technical guide for Mul(Const, Variable) optimization |
| `rust_benchmarks/context.md` | SparseDifferentiation/DiffEngine background |

---

## Appendix: JSON Output Schema

The `--json` flag produces a structured file with this schema:

```json
{
  "metadata": {
    "suite_version": "1.0.0",
    "environment": {
      "python_version": "3.12.3",
      "numpy_version": "2.4.3",
      "scipy_version": "1.17.1",
      "cvxpy_version": "1.8.0.dev0",
      "blas": "scipy-openblas 0.3.31.dev",
      "cpu_brand": "AMD Ryzen 5 7600X 6-Core Processor",
      "cpu_count": 12,
      "architecture": "x86_64",
      "os": "Linux",
      "rayon_num_threads": "auto",
      "omp_num_threads": "1",
      "git_hash": "866f39957",
      "cvxpy_rust_available": true
    },
    "config": {
      "mode": "quick",
      "layers": "bm",
      "backends": ["RUST", "SCIPY"],
      "seed": 42,
      "single_thread": false,
      "sizes": "small,medium"
    }
  },
  "results": [
    {
      "problem": "dense_matmul (n=50)",
      "category": "A: Arithmetic",
      "size_label": "small",
      "dominant_ops": ["dense_const", "mul", "sum_entries"],
      "measurements": {
        "build_matrix": {
          "RUST": {
            "mean_ms": 0.09,
            "median_ms": 0.085,
            "std_ms": 0.012,
            "min_ms": 0.078,
            "max_ms": 0.11,
            "ci_95": [0.075, 0.105],
            "cv": 0.133,
            "n_samples": 5,
            "raw_ms": [0.09, 0.085, 0.11, 0.078, 0.087]
          },
          "SCIPY": { "..." : "..." }
        }
      },
      "speedups": {
        "build_matrix": {
          "RUST_vs_SCIPY": 2.05
        }
      }
    }
  ],
  "scaling_analysis": {
    "constraint_count": {
      "title": "Constraint Count Scaling",
      "axis_name": "n_constraints",
      "axis_values": ["4", "10", "50", "100", "500", "1000", "2000", "5000"],
      "build_matrix_ms": {
        "RUST": [0.18, 0.38, 1.42, 1.14, 3.59, 5.99, 17.55, 34.37],
        "SCIPY": [1.22, 2.70, 17.16, 28.95, 118.87, 233.44, 452.46, 1106.50]
      },
      "speedups": [6.71, 7.20, 12.07, 25.40, 33.14, 38.99, 25.78, 32.20]
    }
  },
  "summary": {
    "total_problems": 27,
    "geometric_mean_speedup": 3.562,
    "arithmetic_mean_speedup": 7.791,
    "min_speedup": 0.729,
    "max_speedup": 42.407,
    "RUST_wins": 26
  }
}
```
