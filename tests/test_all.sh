rm /tmp/openfed.sharefile.test_country
rm /tmp/openfed.sharefile.test_functional

pytest -k 'not leader and not follower'

pytest -k 'leader'  &

pytest -k 'follower'  &

wait

