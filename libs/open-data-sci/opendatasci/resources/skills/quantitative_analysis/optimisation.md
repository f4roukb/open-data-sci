# Quantitative Analysis — Optimisation

- Many quantitative problems can be cast as optimisation; recognising the structure (convex vs. non-convex, constrained vs. unconstrained, continuous vs. integer) determines what solvers are applicable and what guarantees are available
- Convex problems have the substantial advantage that local optima are global optima; non-convex problems may require heuristics, multiple starting points, or relaxations
- Numerical stability matters: poorly conditioned problems can produce results that look precise but are sensitive to small perturbations in inputs or solver tolerances
- In practice, regularisation in optimisation and regularisation in statistics are the same idea expressed in different languages — both bias a solution toward simpler structure in exchange for reduced variance

## Metadata

- parent domain: quantitative_analysis
