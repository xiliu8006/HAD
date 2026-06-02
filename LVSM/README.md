# Bundled LVSM Runtime

This directory contains the LVSM runtime modules used by DreamAware3D for
image and confidence guidance. Training datasets, experiment logs, wandb state,
and checkpoints are not included.

Put LVSM checkpoint files under:

```text
LVSM/checkpoints/LVSM_decoder_only_conf_Resi_unet_512/
```

The default DreamAware3D launcher uses this bundled copy. Set `LVSM_ROOT` only
when you want to point to a different LVSM checkout.
