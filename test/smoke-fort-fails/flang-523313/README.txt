Unfortunately this is technically not allowed in the OpenMP spec.
Mapping of Fortran polymorphic types is not allowed in OpenMP.
Since type bound procedures required passed variables to be declared 
polymophic (CLASS) this prevents using type bound procedures effectively.

The OpenMP committee is actively discussing how to standardize allowing this.
