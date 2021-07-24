# Test

## Test functional units

```bash
pytest -k 'not leader and not follower'
```

## Test leader and follower units

```bash
# Launch leader first
pytest -k 'leader'

# Launch follower
pytest -k 'follower'
```