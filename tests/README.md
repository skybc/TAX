# Test Runner Scripts

This directory contains scripts for running tests.

## Usage

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Performance tests
pytest -m performance

# Slow tests (excluded by default)
pytest -m slow
```

### Run Tests with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Tests in Parallel
```bash
pytest -n auto
```

### Run Specific Test Files
```bash
pytest tests/test_mask_utils.py
pytest tests/test_statistics.py
pytest tests/test_integration.py
```

### Run Tests with Verbose Output
```bash
pytest -v
pytest -vv  # Extra verbose
```

### Generate Test Report
```bash
pytest --html=report.html --self-contained-html
```

## Test Organization

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── test_mask_utils.py          # Mask utilities tests
├── test_statistics.py          # Statistics module tests
├── test_data_manager.py        # Data manager tests
├── test_annotation_manager.py  # Annotation manager tests
├── test_models.py              # Model tests
├── test_integration.py         # Integration/workflow tests
└── test_performance.py         # Performance benchmarks
```

## Test Markers

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (workflows)
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.slow` - Slow tests (> 5 seconds)
- `@pytest.mark.requires_gpu` - Tests requiring CUDA
- `@pytest.mark.requires_sam` - Tests requiring SAM weights
- `@pytest.mark.ui` - UI component tests

## Requirements

```bash
pip install pytest pytest-cov pytest-xdist pytest-html pytest-benchmark
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - run: pip install -r requirements.txt
      - run: pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Test Data

Test data is automatically generated using fixtures defined in `conftest.py`. No manual test data preparation needed.

## Tips

1. **Run unit tests frequently** during development
2. **Run integration tests** before committing
3. **Run performance tests** periodically to catch regressions
4. **Check coverage** regularly to ensure adequate testing
5. **Use markers** to selectively run test subsets
