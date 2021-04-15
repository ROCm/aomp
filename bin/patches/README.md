AOMP bin/patches directory README.md
====================================

This directory contains patches to components that the aomp developers do not have direct control over.
These are referred to as non-AOMP components. non-AOMP components are required to be built for AOMP.
Sometimes changes to these components are necessary to be compatible with aomp.
The patch process described here allows AOMP developers to use exact sources for these components 
without creating a mirror or a separate branch.
If we created a mirror, we would not get maintenance unless we kept updating the mirror.

The components that AOMP developers have direct control over are: 
  
   llvm-project, aomp-extras, flang, aomp

All other components are non-AOMP components that often require patching to
work with AOMP. The AOMP build scripts for non-AOMP components use this patching process:

 * Component patches are applied automatically BEFORE the cmake command runs for a fresh build. This allows patches to fix cmake files.
 * The build script then runs cmake and make commands.
 * The developer or the master script (build_aomp.sh) then runs the component build script with the "install" option.
 * The build script with "install" reverses the patch after successful installation of the component. 

The two supporting bash functions used by the build scripts for the above process are patchrepo and removepatch.
These functions are defined in aomp_common_vars.
These functions take a single argument which is the directory to be patched.
This argument is root directory of the component to patch. If the control file has no patches for the component,
nothing will be applied.

The environment variable AOMP_PATCH_CONTROL_FILE defines the patch control file.
The default value is "patch-control-file.txt".
The patches that this control file points to must be in the same directory as the control file.
The function patchrepo first checks to see if patch was already applied.
If so, it continues without applying the patch.
After checking if patch was already applied, patchrepo tests to see if patch will apply before applying it.
If it will not apply, patchrepo causes the build script to have a fatal fail.
It is not enough to just put a patch file in this directory.
You must also update the patch-control-file.txt file.
We are trying to define a single patch file is used for each repo.
It is easy to build a single patch file for the entire repo with the git diff command.
However, multiple patch files are supported.

## Why do we remove patches after installation?

This allows developers to pull updates for non-AOMP components.
This is typically done with the execution of the clone_aomp.sh script.
If we left the component patched, pull operations could fail.

## What about the composite source tarball?  Is this patched?

Yes, it is patched.  The script create_release_tarball.sh uses aomp_common_vars and applies 
each patch then reverses the patch when complete. 

## Updating an Existing Patch

Sometimes patches get old and they apply but not cleanly.
You will see build log messages such as this.

```
patching file src/fmm.c
Hunk #1 succeeded at 1491 (offset 6 lines).
patching file src/topology.c
Hunk #3 succeeded at 448 (offset 7 lines).
Hunk #4 succeeded at 462 (offset 7 lines).
Hunk #5 succeeded at 838 (offset 2 lines).
Hunk #6 succeeded at 872 (offset 2 lines).
Hunk #7 succeeded at 966 (offset -1 lines).
Hunk #8 succeeded at 1277 (offset -1 lines).
Hunk #9 succeeded at 1317 (offset -1 lines).
```
This occurs when updates were made to the component source AFTER the patch was created.
This is the process to get a fresh new patch for a particular component.
First, run just the component build script without install.  This is running the
script with no options. Then cd to the directory where the repo is located.
Then run 'git status' to verify that the build applied the patch and changes
are present.  Then run 'git diff' to generate a new patch.  Lastly, test
the reversal of the new patch by installing the component.  Then run 'git status'
to verify there are no changes to the non-AOMP component.

Lets use build_roct.sh as an example to create a fresh patch for the roct component.
Before starting,  be sure you are on the development branch of aomp repo
which is currently amd-stg-openmp.  For this demo assume your aomp component repos
are stored at $HOME/git/aomp13.  Run these commands:

```
$HOME/git/aomp13/bin/build_roct.sh
$HOME/git/aomp13/aomp/bin/build_roct.sh
cd $HOME/git/aomp13/roct-thunk-interface
git status
git diff >$HOME/git/aomp13/aomp/bin/patches/roct-thunk-interface.patch
$HOME/git/aomp13/aomp/bin/build_roct.sh install
git status
```
Lastly, please push the updated patch into the aomp development branch
for other developers to pick up with these commands:

```
cd $HOME/git/aomp13/aomp
git add bin/patches/roct-thunk-interface.patch
git commit -m "freshen patch for roct"
git push
```
