# Maximally Entangled Atomic Orbitals (MEAO) for Bonding Analysis
![meao](meao-logo.png)

This is a package for covalent bonding analysis using maximally entangled atomic orbitals (MEAO), and orbital entanglement. We included two example molecules for both two-center bonding and multi-center bonding analysis.

# Dependency
```Python
numpy
scipy
pyscf
block2
```

# Example Usage
Run example files in the parent directory via
```python -m MEAO.examples.n2```. These scripts should only takes a few seconds to execute. In the output one can see the optimization of the MEAO cost function, bonds (MEAO pairs) identified by orbital clustering, a DMRG calculation using the MEAO basis, and a subsequent calculation of orbital-orbital mutual information. Any bonds with significant correlation is printed at the end of the script with orbital indices and correlation value.

# Reference
L. Ding, E. Matito, C. Schilling, arXiv preprint, arXiv:2501.15699, 2025
