# OpenFed

openfed.aggregator: 负责将收集到的模型，按照各种不同的方式，聚合在一起，并计算出最终的梯度赋值给grad属性。
openfed.allocator: 负责任务的分发，响应请求。
openfed.data: 提供在仿真实验中针对不同数据集的处理与划分方式。
openfed.federated: 负责联邦学习通讯的核心模块。
openfed.optimizer: 为联邦学习设计的优化算法。
openfed.backend: 如果该结点被设置为联邦学习的服务中心，那将进入该模块，并且该模块之后的程序不再运行。如果不是联邦学习的服务中心，则不产生任何操作。
openfed.launch: 用于仿真训练时，方便快捷的启动脚本。


客户端要做的事情：
1. 开始训练前：加入一个federated_world.
2. 每一个epoch开始时：向服务器拉取参数，包括模型参数与训练参数。
3. 每一个epoch结束时：向服务器推送参数，包括模型参数与训练参数。
4. 结束训练前：退出所属的federated_world.

服务器要做的事情：
1. 定义好模型model
2. 选择合适的客户端的优化器optim
3. 设置合适的聚合方式aggre
4. 根据模型、优化器和聚合方式，确定需要同步的数据，一般是model.state_dict() + optim.state_dict() + aggre.state_dict()
5. 进入大循环：
   1. 由allocator扫描配置文件，维护与客户端的连接，yield生成器返回对应的任务类型。
   2. 根据任务类型，进行调度：
      1. 调用federated进行模型的同步、收发？
      2. 调用aggregator和optimizer将接收到的模型进行聚合、更新？
   3. 将相关任务设置成完成状态。
   4. 回调函数的处理


数据传递规则：
   客户端和服务器端必须保持相同的state_dict()。这个必须是一致的！
   客户端返回数据时，会以state_dict().keys()作为依据，和服务器沟通所要传递的内容。
   服务器会读取这个列表，然后创建相应的tensor去接收参数。
   deliver_content:
      [key, [tensor_type, tensor_size], ..., ], tensor_type in [int, long, float, float64]
   