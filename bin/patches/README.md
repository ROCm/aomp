AOMP bin/patches directory README.md
====================================

This directory contains patches to components that the aomp developers do not have direct control over.
These are referred to as non-AOMP components. non-AOMP components are required to be built for AOMP.
Sometimes changes to these components are necessary to be compatible with aomp.
The patch process described here allows AOMP develoeprs to use exact sources for these components without creating a mirror.
If we created a mirror, we would not get maintenance unless we kept updating the mirror.
The components that AOMP developers have direct control over are llvm-project, aomp-extras, and aomp.
Changes to non-AOMP must be patched as part of the build process.

Patches are applied BEFORE the cmake in the build script.
Patches are removed after installation. 
The supporting patch bash functions are defined in aomp_common_vars.
Those bash functions are patchrepo, removepatch, and getpatchlist. 
The build script must set variables patchdir and patchloc before calling these functions.
These functions use the control file in this directory called patch-control-file.txt . 
The function patchrepo first checks to see if patch was already applied and if so, it continues without applying patch.
After checking if patch was already applied, patchrepo tests to see if patch will apply before applying it. 
If it will not apply, patchrepo causes the build script to fail.  

It is not enough to just put a patch file in this directory.  You must also update the patch-control-file.txt file.

Why do we remove patches after installation?  This allows developers to pull updates for non-AOMP components. 
