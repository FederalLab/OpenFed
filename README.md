# OpenFed

## 跑一个最简单的demo，来快速理清底层的调用关系：

```bash
(higgs)  ✘ densechen@C02DW0CQMD6R  ~/code/OpenFed   main ±  python client.py
Connect to Server...
>>> Connect... <<<  <openfed.utils.types.FedAddr object at 0x7fa5183dc280>
[W ProcessGroupGloo.cpp:559] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
+++ Connected +++  <openfed.utils.types.FedAddr object at 0x7fa5183dc280>
python(67768,0x70000ca41000) malloc: can't allocate region
:*** mach_vm_map(size=8863084066665136128, flags: 100) failed (error code=3)
python(67768,0x70000ca41000) malloc: *** set a breakpoint in malloc_error_break to debug
xxx Exited xxx  Time: 2021-06-16 21:28:54.964902, File: /Users/densechen/code/OpenFed/openfed/federated/monitor/monitor.py, Func: run, Line: 64
Try to destroy all process group in federated world.
PASS
```

```bash
(higgs)  densechen@C02DW0CQMD6R  ~/code/OpenFed   main ±  python server.py
Connect to Client...
>>> Connect... <<<  <openfed.utils.types.FedAddr object at 0x7fdae05bd280>
[W ProcessGroupGloo.cpp:559] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
+++ Connected +++  <openfed.utils.types.FedAddr object at 0x7fdae05bd280>
xxx Exited xxx  Time: 2021-06-16 21:28:55.002473, File: /Users/densechen/code/OpenFed/openfed/federated/monitor/monitor.py, Func: run, Line: 64
Try to destroy all process group in federated world.
xxx Exited xxx  Time: 2021-06-16 21:28:55.005551, File: /Users/densechen/code/OpenFed/openfed/federated/federated.py, Func: process_generator, Line: 358
PASS
xxx Exited xxx  Time: 2021-06-16 21:28:55.017701, File: /Users/densechen/code/OpenFed/openfed/federated/federated.py, Func: run, Line: 176
```