from datetime import datetime


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


openfed_class_fmt = "\033[0;34m<OpenFed>\033[0m \033[0;35m{class_name}\033[0m\n{description}\n"
