# Tests

This directory contains unit and integration smoke tests.

## Files

- `test_retention.py` - Retention decay and recurrent-state shape checks.
- `test_aligned_architecture.py` - Engram, Block AttnRes, milestone snapshot,
  full model, and toy training-step checks.
- `conftest.py` - Test import path setup.

Run all tests with:

```bash
python -m pytest
```
