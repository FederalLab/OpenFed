from openfed.common.hook import Hook


def test_hook():
    hook = Hook()

    assert len(hook.hook_dict) == 0
    assert len(hook.hook_list) == 0

    hook.register_hook("dict", lambda: "dict")
    assert hook.hook_dict

    def func(): return "list"
    hook.register_hook(func)
    assert len(hook.hook_dict) == 1

    hook.remove_hook("dict")
    assert len(hook.hook_dict) == 0

    hook.remove_hook(func)
    assert len(hook.hook_list) == 0
