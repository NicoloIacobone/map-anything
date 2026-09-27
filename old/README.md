# old/

Code kept for the record, not used by the current version.

Nothing in here is imported by the active pipeline
(`scripts/distill.py` -> `mapanything/distill/distillation.py`), and none of it is
maintained. It is here because it documents how the distillation setup was arrived at.

| path | what it was | why it is here |
|---|---|---|
| `distillation_backup.py` | a 3000-line single-file version of the whole training loop | superseded by `mapanything/distill/distillation.py`; still referenced by the archived jobs in `slurm/old/` |
| `losses_backup.py` | a copy of `mapanything/train/losses.py` | nothing imports it; `SemanticConsistencyLoss` lives in `mapanything/train/losses.py` |
| `sam2_builder_pre_highres.py` | earlier copy of the SAM 2 builder | predates the `use_high_res` switch; the current one is `sam2_minimal/sam2_builder.py` |
| `segmentation_head.py` | driver for a trainable instance head | dead: it imports `nico.my_head`, which is in `old/nico_old/my_head.py` |
| `nico_old/` | first-generation experiments, feature extractors, PCA tests | exploratory, superseded |
| `distillation_scripts/` | roughly 20 intermediate versions of the training script | the iteration history of the distillation loop |

`old/nico_old/` and `old/distillation_scripts/` contain many near-duplicate files.
They were not deduplicated on purpose: the differences between them are the record of
what was tried.
