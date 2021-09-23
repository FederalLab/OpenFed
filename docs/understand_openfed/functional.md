## Functional

There are three mainly kinds of hooks, i.e., package hook, unpackage hook and step hook.
All these hooks can be automatically register to a maintainer in `with maintainer` context.
There is a `nice` value to control the order of the hooks to apply. A lower `nice` value means a higher priority.

### Step

`Step` hook is mainly used for control aggregator operations. You can define a step hook and register it to a maintainer via :func:`register_step_hook`.

### Package and Unpackage

`Package` and `Unpackage` hooks usually pair up with each other. This hook is used for pack data before upload and unpack data after download. You can define a package hook and register it to a maintainer via :func:`register_package_hook`. You can also define a unpackage hook and register it to a maintainer via :func:`register_unpackage_hook`.
