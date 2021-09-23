## Core

### Maintainer

:class:`Maintainer` bridges the connection between upper(federated algorithms) and lower(communication and topology) layers. It has the following properties:

- `pipe`: The currently target to communicate with. A maintainer will manage several pipes in the same time, and `pipe` will indicate what is the current target.
- `pipes`: A list of pipes to communicate with.
- `current_step`: It is used to indicate which step is running on.
- `fed_props`: Actually, a maintainer is corresponding to a specified federated group. We record the related federated group properties in this attributions.

You can use :class:`Maintainer` to conduct a flexible communication with other nodes more easily compared with :class:`Pipe`.
