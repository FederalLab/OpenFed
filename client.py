from openfed import Frontend, FedAddr


# build a connect
frontend = Frontend(fed_addr=FedAddr(
    backend="gloo", init_method='tcp://127.0.0.1:33298'))
