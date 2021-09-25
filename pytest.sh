# If you add new test cases, remember copy it to `.github/workflows/build.yml`
echo 'General test...'
sleep 1
pytest -k 'not aggregator and not collaborator'

echo 'Federated...'
sleep 1
pytest -n 3 tests/test_federated/test_federated.py -k 'federated'

echo 'Maintainer...'
sleep 1
pytest -n 3 tests/test_core/test_maintainer.py -k 'maintainer'

echo 'Simulator...'
sleep 1
pytest -n 3 tests/test_simulator.py -k 'simulator'

echo 'Paillier Crypt...'
sleep 1
pytest -n 2 tests/test_paillier_crypto.py -k 'paillier_crypto'
