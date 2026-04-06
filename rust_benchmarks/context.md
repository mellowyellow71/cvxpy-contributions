some updates on this (only 1.5 years later ) I have been experimenting with speeding up canonicalization by re-using the derivative computation from the diffengine. TLDR: I was able to reduce the compile time for this least-squares problem down to 3.7s CPP backend is 9.7s SCIPY is 11.8s COO is 14.8s The original problem is: minimize |Ax-b|_2^2 The canonicalized problem is: minimize t^Tt s.t. t=Ax-b The canonical variable is y = [t; x] of dimension m + n. The matrix stuffing is relatively straightforward:
 P is (m+n) × (m+n):
 P = [ 2I_m 0 ]
 [ 0 0 ]
 A partial identity — 2I on the m auxiliary vars, zeros on the n original vars. This
 encodes Σ tᵢ².
 AF is m × (m+n):
 AF = [ -I_m | A ]
 Block structure: a negative identity on the auxiliary block, then the dense A on the original block. The equality AF @ y + bg == 0 gives -t + A @ x = b, i.e., t = A @ x - b.

Some more context that could be helpful: this PR added a dense matrix class https://github.com/SparseDifferentiation/SparseDiffEngine/pull/49 which helped form and pass the A directly as a dense matrix. then this PR https://github.com/SparseDifferentiation/SparseDiffEngine/pull/51 added a fast path which uses BLAS copies for sparse row vectors with one entry (when evaluating the expression). This happens because in the example above the expensive expression tree to evaluate is: mul(A,x), and its jacobian performs A @ I_x (identity of dimension x) It is a bit silly because in this example we could just return A directly.. but we don't want to special case it. I made a branch which adds a diffengine backend: https://github.com/cvxgrp/DNLP/pull/180 in case anyone wants to have a look. (you would also need to install the latest sparsediffengine) please let me know if there are questions on that end. For the quadratic part of the objective, we can take the hessian using the diffengine. And since it is a single SymbolicQuadForm, it is able to compute the correct P.
On a related note, I also opened a PR to cvxpy master: https://github.com/cvxpy/cvxpy/pull/3240. After digging into this example further, I noticed that there were some potential inefficiencies in certain QP solver interfaces. Firstly, we are performing row slicing to split up AF into A and F, however since the original matrix is in CSC (from the apply_parameters call), it is taking a bit of time to do this slicing (it is preferred to do it in CSR). Secondly, there were some unnecessary copies and format conversions being made at certain places. Finally, in OSQP for example, we would anyways end up vstacking A and F back into AF (this was because we wanted to negate the F to be conformant to OSQP's standard form). Since A and F are in CSC, once again vertically stacking them took a bit of time (this would be much faster in CSR, since we just need to concatenate and apply an offset to the row pointers). These operations are all taking some time for this least-squares example (which admittedly, has 80M + nonzeros).. and I presume it could be the case also for very large problems. Claude noticed a really nice improvement, which was to not do any such splitting and only change the sign of the bounds. I need to double check this, but it seems to be correct mathematically and also very elegant (although it requires to also change the sign of the corresponding dual variables). Please let me know if this is something we can consider adding.. and if it could cause some "breaking" changes potentially (I hope not). We would eventually probably need to go over all QP interfaces and see if they could benefit from this as well.



P.S. for anyone curious
claude generated some instructions for testing this:
```py
  ---                                                                                      
  Prerequisites: Python >= 3.11, a C++ compiler (for building extensions)                  
                                                                                           
  1. Install SparseDiffPy (v0.1.5+)                                                        
  git clone --recursive https://github.com/SparseDifferentiation/SparseDiffPy.git          
  cd SparseDiffPy                                                                          
  pip install -e .                                                                         
  cd ..                                                                                    
                                                                                           
  2. Install DNLP fork of CVXPY                                                            
  git clone https://github.com/cvxgrp/DNLP.git
  cd DNLP                                                                                  
  git checkout diffengine-extractor-v2                                                     
  pip install -e .                    
                                                                                           
  3. Run the experiment                                                                  
  python cvxpy/tests/remove_later.py
```






