# tests/

Standalone check scripts, not a pytest suite. Run each one directly:

```bash
python tests/test_sam_decoder.py
```

Each inserts the repository root on `sys.path`, so they work from any working directory.

| script | what it checks |
|---|---|
| `test_amp.py` | mixed-precision (bf16) behaviour of the model |
| `test_nico_sam.py` | the SAM 2 teacher feature extractor against MapAnything |
| `test_sam_decoder.py` | that the vendored SAM 2 mask decoder, prompt encoder and transformer build and run |

They need GPU, model weights and dataset paths that are not in this repository, so they
are not runnable on a clean checkout. See the DA VERIFICARE list in the README.
