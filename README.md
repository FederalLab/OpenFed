# OpenFed

## 跑一个最简单的demo，来快速理清底层的调用关系：

```bash
python server.py
    ########################################
    [W ProcessGroupGloo.cpp:559] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
    Try to destroy all process group in federated world.
    Existed via safe_exited().
    Test success of backend.
    Existed via safe_exited().
    Existed via safe_exited().
```

```bash
python client.py
    [W ProcessGroupGloo.cpp:559] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
    ****************************************
    Try to destroy all process group in federated world.
    Test success of frontend.
    Existed via safe_exited().
```