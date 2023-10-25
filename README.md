# Propeller Gmsh

Propeller Gmsh is a toolkit aimed a generating high-order partially structured meshes around simple geometries of rods, airfoils and propeller blades. It is based on the opensource library Gmsh ([gmsh.info](https://gmsh.info/)). Mesh parameters can be controlled at will and the mesh can be made fully hex.

`gmshToolkit.py` contains the basic routines to generate the mesh. `gmshRodAirfoil.py`, `gmshRodAirfoil_2D.py` and `gmshPropeller.py` provide illustrations of the use of the toolkit in 2D, 2.5D and 3D respectively. The grid close to the rod/airfoil surfaces and in the airfoil wake are structured and unstructured elsewhere.

`yPlus_estimate.py` compares different empirical friction coefficient laws from the literature to estimate the mesh y+ for the targeted application.

`gmshSphere.py` and `gmshCylinder.py` are used to generate structured/unstructered spherical and cylindrical control surfaces (data interpolation).

`highOrderMeshing.py` is a script to generate a minimalistic 2D high-order mesh to test mesh Gmsh's algorithms.

These scripts run with python 3 and assume gmsh-api is installed, e.g. using, 

```
pip install gmsh
```

To enable the parallel version of the gmsh-api, follow: https://gitlab.onelab.info/gmsh/gmsh/-/issues/1422

Bug tracking and sugestions are welcome.
For general documentation, refer to: 
   + https://gmsh.info/doc/texinfo/gmsh.html
   + https://bthierry.pages.math.cnrs.fr/tutorial/gmsh/
