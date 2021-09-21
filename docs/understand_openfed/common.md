## Common

### Meta

`Meta` class is a special dictionary that used to convey messages between aggregator and collaborators. It contains two default attributions:

- `mode`: String in [`train`, `others`]. If `mode==train`, the collaborator will train the global model with personal privacy data, and upload the trained model to aggregator. The aggregator will automatically aggregate the received models. Otherwise, the collaborator will not update the global model. It will test the global on personal privacy data and return the results to aggregator.
- `version`: Int. The version number of received global model. In federated learning, we need to use this version tag to control the update behavior of aggregator. Sometimes, the aggregator receives the invalid version of model, which may be out of date. When this case occurs, aggregator will apply some tragedies to deal with it.

`Meta` class can used as a standard dictionary:

```shell
>>> import openfed
>>> meta = openfed.Meta()
>>> meta
<OpenFed> Meta
+-------+---------+
|  mode | version |
+-------+---------+
| train |    -1   |
+-------+---------+

>>> meta['timestamp'] = openfed.utils.time_string()
>>> meta
<OpenFed> Meta
+-------+---------+---------------------+
|  mode | version |      timestamp      |
+-------+---------+---------------------+
| train |    -1   | 2021-09-21 09:50:41 |
+-------+---------+---------------------+
```

`Meta` class can also be used as a class to access his attributions:

```shell
>>> import openfed
>>> meta = openfed.Meta()
>>> meta
<OpenFed> Meta
+-------+---------+
|  mode | version |
+-------+---------+
| train |    -1   |
+-------+---------+

>>> meta.timestamp = openfed.utils.time_string()
>>> meta
<OpenFed> Meta
+-------+---------+---------------------+
|  mode | version |      timestamp      |
+-------+---------+---------------------+
| train |    -1   | 2021-09-21 09:52:47 |
+-------+---------+---------------------+
```

### Address

`Address` class stores all the arguments needed to build a process group. It will automatically check the arguments you passed in. There are two kinds of address:

- `tcp_address`: TCP address will keep the communication via a tcp address.
- `file_address`: File address will keep the communication via a shared file.

We also provide an `empty_address`, which contains nothing information, to play as a placeholder.

Define a tcp address:

```shell
>>> import openfed
>>> tcp_address = openfed.Address('gloo', 'tcp://localhost:1994')
>>> tcp_address
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | tcp://localhost:... |     2      |  -1  |
+---------+---------------------+------------+------+
```

Load the `default_tcp_address`:

```shell
>>> import openfed
>>> openfed.default_tcp_address
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | tcp://localhost:... |     2      |  -1  |
+---------+---------------------+------------+------+
```

Define a file address:

```shell
>>> import openfed
>>> file_address = openfed.Address('gloo', 'file:///tmp/openfed.sharedfile')
>>> file_address
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | file:///tmp/open... |     2      |  -1  |
+---------+---------------------+------------+------+
```

Load the `default_file_address`:

```shell
>>> import openfed
>>> openfed.default_file_address
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | file:///tmp/open... |     2      |  -1  |
+---------+---------------------+------------+------+
```

Load the `empty_address`:

```shell
>>> import openfed
>>> openfed.empty_address
<OpenFed> Address
+---------+-------------+------------+------+
| backend | init_method | world_size | rank |
+---------+-------------+------------+------+
|   null  |     null    |     2      |  -1  |
+---------+-------------+------------+------+
```

You can refer to the API documentation for more details about each arguments.
