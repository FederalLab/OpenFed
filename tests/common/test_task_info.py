from openfed.common.task_info import TaskInfo

def test_task_info():
    task_info = TaskInfo()
    task_info.instances = 128
    assert task_info.instances == 128

    print(task_info.info_dict)

    print(task_info)

test_task_info()