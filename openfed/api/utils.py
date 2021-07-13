def download_callback(api):
    task_info = api.delivery.task_info
    api.delivery_task_info = task_info

    # unpack state
    if api.frontend:
        [api.unpack_state(pipe) for pipe in api.pipe]
    elif api.backend:
        # Increase the total number of received models
        api.received_numbers += 1
        packages = api.tensor_indexed_packages
        [container.step(packages, task_info) for container in api.container]
