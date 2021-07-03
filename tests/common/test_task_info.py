from openfed.common.task_info import TaskInfo

def test_task_info():
    task_info = TaskInfo()
    task_info.set("instances", 128)
    assert task_info.get("instances") == 128

    print(task_info.as_dict)

    print(task_info)