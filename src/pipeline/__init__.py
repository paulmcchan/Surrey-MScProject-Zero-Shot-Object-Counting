"""
Reusable pipeline stages for the MSc zero-shot counting project.

Modules
-------
io_cache : read/write the project's cache formats (B0 masks, A3c masks,
           Marigold depth, M1 split labels).
sg       : SG counting logic (SG_v1 frozen behaviour, SG_v1.1 fix).
b0       : unguided SAM2 automatic mask generation (frozen B0 setup).
semantic : SAN target score maps and the A3c semantic mask.
depth    : Marigold relative depth (frozen M1 setup).
"""
