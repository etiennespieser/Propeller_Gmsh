# Propeller Gmsh

Propeller Gmsh is a toolkit aimed a generating meshes around propeller blades. It is based on the opensource library Gmsh ([gmsh.info](https://gmsh.info/)). The mesh close to the blade and in the airfoil wake is structured and unstructured elsewhere. Mesh parameters can be controlled at will.

`gmshToolkit.py` contains the basic routines to generate the mesh. `gmshAirfoil.py` and `gmshPropeller.py` provide to illustrations of the use of the toolkit in 2D and 3D respectively.

These scripts run with python 3 and assume gmsh-api is installed, e.g. using, 

```
pip install gmsh
```

Bug tracking and sugestions are welcome. For general documentation, refer to: https://gmsh.info/doc/texinfo/gmsh.html
