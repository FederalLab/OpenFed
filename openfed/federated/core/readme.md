# 模块说明
在这个模块里面，所有的函数和类，都是以联邦世界做为基本对象，编写的函数。
换言之，在这里还未区分出king和queen之间的区别。

## functional.py
    包含了用于数据交换的函数，例如all_reduce, gather, scatter。
    这些函数拓展了原始相对应的pytorch函数，使得其能够在指定的联邦世界中进行安全的数据交换。
    为了更好的隔绝与pytorch中distributed库的冲突，在这里我们重新实现了相关函数，而不是采取调用的方式。

## federated_c10d.py
    封装实现了Federated World。它包含了创建、销毁、管理各个联邦世界的连接。