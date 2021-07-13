def download_callback(api):
    task_info           = api.delivery.task_info
    api.delivery_task_info = task_info

    # unpack state
    if api.frontend:
        [api.unpack_state(ft_opt) for ft_opt in api.ft_optimizer]
        if api.pipe is not None:
            [api.unpack_state(pipe) for pipe in api.pipe]
    elif api.backend:
        # Increase the total number of received models
        api.received_numbers += 1
        if api.reducer is not None:
            [reducer.step(task_info) for reducer in api.reducer]
        packages = api.delivery.tensor_indexed_packages
        [aggregator.step(packages, task_info) for aggregator in api.aggregator]
