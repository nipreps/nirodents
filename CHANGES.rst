0.2.6 (March 25, 2022)
======================

  * CI: docker deployment (#56)
  * pin: bump niworkflows to 1.5.x (#55)

0.2.5 (March 15, 2022)
======================

  * FIX: N4 updates (#54)
  * FIX: input image absolute path (#53)
  * FIX: update brain extraction notebook (#51)
  * FIX: reorder pre-brain extraction massaging
    PIN: pandas and scipy versions in docker file (#49)
  * PIN: niworkflows 1.4.x (#48)
  * DOC: update changes.rst (#47)

0.2.3 (September 15, 2021)
==========================

  * ENH: improved clip function (#45)
  * ENH: improved brain extraction parameters (#44)
  * FIX: cli updated so antsai no longer default (#42)
  * ENH: improve aniso bspline (#41)
  * ENH: scale Laplacian smoothing with voxel size (#40)
  * ENH: Continue refactoring the workflow (#38)
  * REL: Preparing a 1.0 release (#37)
  * ENH: adapt antsai paramaters from cli (#36)
  * ENH: Deep revision of the workflow (#35)
  * ENH: Add RATS (commented out) and PCNN to dockerfile (#34)
  * ENH: Second refactor of workflow - make ``antsAI`` optional (#33)
  * ENH: Add an entrypoint in container images (#32)
  * ENH: Several improvements over the overhaul (#31)
  * ENH: Workflow overhaul (#30)
  * MAINT: Run black on the full repo, address pep8 errors (#27)
  * MAINT: tidy workflow (#23)
  * ENH: Setup a smoke test on CircleCI + minor improvements to CLI (#26)
  * FIX: Correctly pin niworkflows branch and use new interface (#25)
  * FIX: resampling bug (#22)
  * MAINT: Update version pinning of nipype and niworkflows to dev versions (#20)
  * ENH: Add AFNI to docker image (#19)
  * FIX: Data init file (#18)
  * FIX: Set correct binary path for ANTS (#17)
  * ENH: add mosaic plots and wrapper (#16)

0.2.0 (October 06, 2020)
========================
First usable release, still in alpha status.

* FIX: Correctly pin niworkflows branch and use new interface (#25)
* FIX: Bug in resampling interface (#22)
* FIX: Data init file (#18)
* FIX: Set correct binary path for ANTS (#17)
* ENH: Improve anisotropic B-Splines for INU correction (#41)
* ENH: Scale Laplacian smoothing with voxel size (#40)
* ENH: Continue refactoring the workflow (#38)
* ENH: Adapt ``antsAI`` paramaters from CLI (#36)
* ENH: Deep revision of the workflow (#35)
* ENH: Add RATS (commented out) and PCNN to ``Dockerfile`` (#34)
* ENH: Second refactor of workflow - make ``antsAI`` optional (#33)
* ENH: Add an entrypoint in container images (#32)
* ENH: Several improvements over the overhaul (#31)
* ENH: Workflow overhaul (#30)
* ENH: Setup a smoke test on CircleCI + minor improvements to CLI (#26)
* ENH: Add AFNI to docker image (#19)
* ENH: add mosaic plots and wrapper (#16)
* MAINT: Run black on the full repo, address pep8 errors (#27)
* MAINT: Tidy-up workflow (#23)
* MAINT: Update version pinning of nipype and niworkflows to dev versions (#20)

