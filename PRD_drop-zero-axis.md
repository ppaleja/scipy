# Product Requirements Document (PRD)
Enhancement: SciPy Sparse – add new `drop_zero_axis` method  
Reference: [ENH: sparse: add new drop_zero_axis method](https://github.com/scipy/scipy/issues/6754)

1. Introduction/Summary
- Overview: Introduce a `drop_zero_axis` method to SciPy sparse matrix and array types that removes entirely zero rows or columns efficiently. This is particularly useful in machine learning and data preprocessing pipelines where sparse features or samples may be empty and should be pruned to reduce memory footprint, improve computational performance, and eliminate noisy inputs.
- Purpose: Provide a standardized, efficient, and format-preserving API to drop zero axes across sparse formats (CSR, CSC, COO, BSR, DIA, LIL, DOK), with clearly defined semantics regarding explicit zeros, duplicates, and shape changes. Aligns with existing sparse APIs and patterns (e.g., axis-aware reductions in `_data.py` and in-place zero elimination in `_compressed.py`).

2. Goals
- G1: Implement `drop_zero_axis(axis)` for 2D sparse arrays/matrices, supporting axis=0 (drop zero columns) and axis=1 (drop zero rows).
- G2: Ensure runtime efficiency comparable to format-optimal slicing (CSR for rows, CSC for columns), with internal conversions when necessary and preservation of the output format by default.
- G3: Define explicit, documented semantics with respect to explicit zeros and duplicates, and provide optional parameters to control behavior (e.g., structural zeros vs. stored zeros).
- G4: Provide robust test coverage across key sparse formats (CSR, CSC, COO, BSR, DIA, LIL, DOK) and both array and matrix APIs.
- G5: Maintain backwards compatibility and minimal API surface to reduce maintenance burden.

3. Personas
- Data Scientist Dana (Mid-level)
  - Works on medium-sized datasets (10^5–10^7 entries) with sparse feature matrices.
  - Needs a simple API to prune empty rows/columns before model training to reduce memory and improve speed.
  - Prefers clear defaults and predictable format preservation.

- ML Engineer Eli (Senior)
  - Operates in production pipelines; focuses on throughput and reliability.
  - Needs efficient pruning of zero axes with explicit control over semantics and performance characteristics (e.g., eliminating explicit zeros first).
  - Values stable behavior across formats and good documentation.

- Researcher Rui (Academic/PhD)
  - Experiments with different sparse formats and custom preprocessing.
  - Needs a uniform interface and correctness guarantees; cares about how duplicates and explicit zeros are handled.
  - Wants reproducibility and clear error messages on edge cases.

4. User Stories
- As a data scientist, I want to drop all-zero rows from my sparse training data so that my model trains faster and uses less memory.
- As an ML engineer, I want a format-preserving method to drop zero columns from a sparse matrix so downstream code that expects CSR/CSC continues to work unchanged.
- As a researcher, I want the option to treat explicit stored zeros as zeros to drop, or ignore them, so that I can match the semantics of my experiment.
- As a library user, I want consistent behavior across sparse formats (CSR, CSC, COO, etc.) so that I don’t have to rewrite preprocessing when switching formats.

5. Features

Feature A: `drop_zero_axis` API
- Description: Adds a method to sparse matrix and array types to remove entirely zero rows or columns.
- UI and Inputs (API surface):
  - Method signature (proposed):
    - `drop_zero_axis(axis=0, *, structural=True, copy=True) -> sparse`
      - `axis`: int, {0, 1}. 0 = drop columns; 1 = drop rows.
      - `structural`: bool. If True, treat only structural nonzeros (after canonicalization and optionally eliminating explicit zeros). If False, stored explicit zeros count as “non-zero” and will prevent dropping.
      - `copy`: bool. Return a new sparse object; defaults to True. In-place is out of scope for v1 to avoid complexity (slicing fundamentally creates a new object).
  - Behavior:
    - Computes a mask of non-empty axes using fast per-axis counts.
    - Slices the matrix to drop empty axes using format-optimal paths.
    - Preserves the original format and dtype by default.
- User Flow:
  1) User calls `A.drop_zero_axis(axis=1)` to drop zero rows.
  2) Internally, the method:
     - Validates axis and dimensionality.
     - Optionally canonicalizes (`sum_duplicates()`) and, if `structural=True`, eliminates explicit zeros (`eliminate_zeros()`).
     - Computes counts per axis using format-optimized logic (CSR for rows, CSC for columns) or converts as needed.
     - Builds mask = counts > 0.
     - Performs major-axis slicing to drop zero axes.
     - Returns the result, converted back to original format if any internal conversion occurred.
  3) User receives reduced sparse object and can proceed with downstream tasks.
- Data Collected:
  - None externally; internal computations involve counts per axis and masks. No telemetry or user data is collected.
- Conditions for Success:
  - Correct shape reduction (e.g., (m, n) → (m’, n) or (m, n’)).
  - Content preserved for remaining axes.
  - Output format preserved and dtype unchanged.
  - Runtime comparable to manual axis counting + slicing for large sparse matrices.

Feature B: Semantics Control: Structural vs. Stored
- Description: Allow users to choose how explicit stored zeros affect dropping.
- Inputs:
  - `structural`: bool flag as above.
- User Flow:
  - If `structural=True`: Internally canonicalize indices and remove explicit zeros before counting (via `sum_duplicates()` and `eliminate_zeros()`), ensuring that only truly non-empty axes remain.
  - If `structural=False`: Do not eliminate explicit zeros; use counts that include stored zeros.
- Data Collected:
  - None; internal-only.
- Conditions for Success:
  - Clear, documented behavior.
  - Deterministic outcomes consistent with configuration.

Feature C: Format-Optimized Execution
- Description: Aligns internal logic to CSR/CSC for efficient counting and slicing.
- Inputs:
  - `axis` determines the preferred format. Axis=1 → prefer CSR; Axis=0 → prefer CSC.
- User Flow:
  - If current format optimal: slice directly.
  - Else: convert to optimal format; perform operations; convert back using `asformat(original_format)`.
- Data Collected:
  - None.
- Conditions for Success:
  - Near-optimal performance; limited overhead from conversions.
  - Accurate preservation of format and content.

Feature D: Error Handling and Edge Cases
- Description:
  - Validate axis in {0, 1}; value errors otherwise.
  - Empty shapes: if all rows/cols are dropped, return a (0, n) or (m, 0) sparse object of same format/dtype.
  - Unsupported formats: provide reasonable behavior via conversion or document limitations (e.g., DOK/LIL may convert to CSR/CSC).
- Inputs:
  - `axis`, `structural`.
- User Flow:
  - Errors raised on invalid axis or ndim != 2.
  - Clear, actionable messages.
- Conditions for Success:
  - Predictable and documented behavior across formats.
  - No surprises in corner cases.

6. Constraints
- Technical Constraints:
  - Must work across both matrix API (`spmatrix` and subclasses) and array API (`sparray` and subclasses) without introducing breaking changes.
  - Maintain performance expectations: avoid minor-axis slicing when possible; use CSR/CSC for counting and slicing.
  - DOK/LIL may lack efficient axis-wise operations; conversion to CSR/CSC may be necessary.
  - Explicit zeros semantics must be well-defined; `eliminate_zeros()` is in-place. For structural=True, consider working on a temporary converted copy to avoid mutating the user’s object.
- API Constraints:
  - Keep the API minimal and consistent with SciPy conventions.
  - Avoid introducing in-place “drop” initially; slicing naturally returns a new object.
- Documentation Constraints:
  - Provide examples and clarify explicit vs. structural semantics.

7. Success Metrics
- Adoption: Number of projects/issues referencing `drop_zero_axis` in release cycle +1/+2.
- Performance: Benchmarks showing ≥20% reduction in preprocessing time compared to naive dense operations for typical sparse ML use cases (e.g., dropping empty samples/features).
- Reliability: Test coverage across formats (≥90% lines for the new method), passing CI; zero regressions reported in sparse indexing/conversion tests.
- Usability: Positive feedback via GitHub issues/PRs; minimal confusion about semantics (few documentation-related questions).

8. Future Considerations
- In-place variant: `drop_zero_axis_inplace(axis, structural=True)` with careful guarantees and warnings.
- Multi-axis drop: Combined operation to drop both zero rows and zero columns in one pass with minimal conversions.
- Higher-dimensional support: If future sparse arrays generalize beyond 2D, extend semantics and axis handling accordingly.
- Mask return option: Return the mask of kept axes alongside the pruned matrix for downstream alignment.
- Integration with other preprocessing utilities: Compose with standard normalization, scaling, and sparsity checks.

Appendix: Implementation Guidance (non-binding)
- Location: Implement shared logic at the base level (e.g., `_base.py` for array API and `_matrix.py`/shared mixins) so subclasses inherit behavior.
- Reference Patterns:
  - Axis-aware reduction and mask construction from `_data.py` (tocsc/tocsr, sum_duplicates, minor reduce).
  - Explicit zeros handling via `eliminate_zeros()` in `_compressed.py`, `_coo.py`, `_bsr.py`.
  - Efficient per-axis count in `_compressed.py::count_nonzero`.
- Testing:
  - Parametrize across formats: csr/csc/coo/bsr/dia/lil/dok; both array and matrix types where applicable.
  - Validate shape reduction, format preservation, dtype consistency, and contents after drop with/without structural flag.
  - Large random sparse matrices to check performance and stability.
- Documentation:
  - Add examples in sparse tutorial illustrating use cases (ML feature pruning, dataset cleaning) and structural vs. stored semantics.
