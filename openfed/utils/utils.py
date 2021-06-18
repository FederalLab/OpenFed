from datetime import datetime

def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')