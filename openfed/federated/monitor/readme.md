优化gpu_info, sys_info的信息结构，完善编解码的整个过程。
1. 将GPU、SYS的数据转换成namedtupled。
2. 提供相关的方法，进行数据的处理。

monitor是一个监视器，主要负责监控整个openfed运行的状态。
并且会自动在服务器和客户端之间同步。
如果我们需要上传一些训练过程中的进度或者最终的训练结果，以及openfed的运行状态等信息，
请使用这个模块进行交互。
这个模块会以一个独立的后台进行保持运行。

# TODO
1. 加强Monitor的功能
2. 为GPU INFO创建合适的数据结构，提供GPU INFO到str以及str解析成GPU INFO的能力
3. 同2，为SYS INFO创建一样的工作
4. 负责对接收到的informer数据做进一步的处理，例如解析成task id，results等等，并转换成相应的数据结构。


monitor只负责逻辑层面的代码管理
informor会实现具体的一些功能操作，例如基本信息的收集，等等，并且添加一个时间戳。