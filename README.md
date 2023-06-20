# Propeller Gmsh

Propeller Gmsh is a toolkit aimed a generating meshes around propeller blades. It is based on the opensource library Gmsh ([gmsh.info](https://gmsh.info/)). The mesh close to the blade and in the airfoil wake is structured and unstructured elsewhere. Mesh parameters can be controlled at will and the mesh can be made fully hex.

`gmshToolkit.py` contains the basic routines to generate the mesh. `gmshAirfoil.py`, `gmshRodAirfoil.py` and `gmshPropeller.py` provide three illustrations of the use of the toolkit in 2D, 2.5D and 3D respectively.

These scripts run with python 3 and assume gmsh-api is installed, e.g. using, 

```
pip install gmsh
```

To enable the parallel version of the gmsh-api, follow: https://gitlab.onelab.info/gmsh/gmsh/-/issues/1422

Bug tracking and sugestions are welcome.
For general documentation, refer to: 
   + https://gmsh.info/doc/texinfo/gmsh.html
   + https://bthierry.pages.math.cnrs.fr/tutorial/gmsh/
