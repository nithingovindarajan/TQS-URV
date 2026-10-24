# URV decomposition of SSS and HSS matrices — source papers

## SSS (sequentially semi-separable)
- `Chandrasekaran_et_al_2002_Fast_stable_solvers_SSS.pdf` — S. Chandrasekaran, P. Dewilde, M. Gu, T. Pals, A.-J. van der Veen,
  "Fast stable solvers for sequentially semi-separable linear systems of equations", HiPC 2002 (LNCS 2552), pp. 545–554.
  Introduces SSS matrices; two-sided orthogonal (URV-type) elimination and one-pass Moore–Penrose solver.
- `Chandrasekaran_et_al_2005_Some_fast_algorithms_SSS_SIMAX.pdf` — S. Chandrasekaran, P. Dewilde, M. Gu, T. Pals, X. Sun,
  A.-J. van der Veen, D. White, "Some fast algorithms for sequentially semiseparable representations",
  SIAM J. Matrix Anal. Appl. 27(2):341–364, 2005. doi:10.1137/S0895479802405884

## HSS (hierarchically semi-separable)
- `Chandrasekaran_Gu_Pals_2006_Fast_ULV_solver_HSS_SIMAX.pdf` — S. Chandrasekaran, M. Gu, T. Pals,
  "A fast ULV decomposition solver for hierarchically semiseparable representations",
  SIAM J. Matrix Anal. Appl. 28(3):603–622, 2006. doi:10.1137/S0895479803436652
  Fast, stable implicit ULV solver for HSS, plus a construction algorithm for HSS representations.
- `Dewilde_Chandrasekaran_2006_HSS_Moore-Penrose_URV_solver.pdf` — P. Dewilde, S. Chandrasekaran,
  "A hierarchical semi-separable Moore–Penrose equation solver", Operator Theory: Adv. Appl. 167:69–85, 2006.
  Transforms an HSS representation into a URV (Moore–Penrose) representation.
- `Sheng_Dewilde_Chandrasekaran_2007_Algorithms_to_solve_HSS_systems.pdf` — Z. Sheng, P. Dewilde, S. Chandrasekaran,
  "Algorithms to solve hierarchically semi-separable systems", Operator Theory: Adv. Appl. 176:255–294, 2007.
  Explicit ULV/URV factorization and Moore–Penrose inversion for HSS; relates HSS to SSS.

## Not downloaded (TU Delft servers unreachable at the time) — origin of the URV idea for time-varying systems / SSS
- A.-J. van der Veen, "Time-varying lossless systems and the inversion of large structured matrices",
  AEÜ 49(5/6):372–382, 1995.
- P. Dewilde, A.-J. van der Veen, "Inner–outer factorization and the inversion of locally finite systems of equations",
  Linear Algebra Appl. 313:53–100, 2000. https://sps.ewi.tudelft.nl/pubs/laa00.pdf
- P. Dewilde, A.-J. van der Veen, *Time-Varying Systems and Computations*, Kluwer, 1998 (book).
