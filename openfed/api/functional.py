def download_callback(api):
    api.version = api.reign.upload_version
    task_info = api.reign.task_info
    api.reign_task_info = task_info

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
        packages = api.reign.tensor_indexed_packages
        [aggregator.step(packages, task_info) for aggregator in api.aggregator]
