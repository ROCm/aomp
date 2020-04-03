AOMP bin/patches directory README.md
====================================

This directory contains patches to components that the aomp developers do not have direct control over.
These are referred to as non-AOMP components. non-AOMP components are required to be built for AOMP.
Sometimes changes to these components are necessary to be compatible with aomp.
The patch process described here allows AOMP developers to use exact sources for these components 
without creating a mirror or a separate branch.
If we created a mirror, we would not get maintenance unless we kept updating the mirror.

The components that AOMP developers have direct control over are: 
  
   amd-llvm-project, aomp-extras, flang, aomp

Changes to other non-AOMP components use this  patching process.
Patches are applied BEFORE the cmake in the build script.
Patches are removed after installation of the component. 
The supporting patch bash functions are defined in aomp_common_vars.
Those bash functions are patchrepo, removepatch, and getpatchlist. 
The build scripts only call patchrepo and removepatch with a single argument that is the directory to be patched.

The environment variable AOMP_PATCH_CONTROL_FILE default is patch-control-file.txt.
The patches that this control file points to must be in the same directory as the control file
The function patchrepo first checks to see if patch was already applied.
If so, it continues without applying the patch.
After checking if patch was already applied, patchrepo tests to see if patch will apply before applying it. 
If it will not apply, patchrepo causes the build script to have a fatal fail.  
It is not enough to just put a patch file in this directory.  
You must also update the patch-control-file.txt file.
We are trying to define a single patch file is used for each repo.  
It is easy to build a single patch file for the entire repo with the git diff command. 
However, multiple files are supported.

Why do we remove patches after installation?  This allows developers to pull updates for non-AOMP components. 
