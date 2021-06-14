import os
import platform
import psutil


def getSysInfo() -> dict:
    info = {}
    info['uname'] = platform.uname()._asdict()

    info['cpu_times'] = psutil.cpu_times()._asdict()
    # scputimes(user=650613.02, nice=22.14, system=154916.5, idle=16702285.26, iowait=68894.55, irq=3.38, softirq=7075.65, steal=0.0, guest=0.0)

    info['cpu_percent'] = psutil.cpu_percent(
        interval=None, percpu=True)
    # 每个cpu的使用情况

    info['mem'] = psutil.virtual_memory()._asdict()
    # svmem(total=4018601984, available=1066205184, percent=73.5, used=3904004096, free=114597888, active=3302174720, inactive=426078208, buffers=156520448, cached=795086848)
    # 其中percent表示实际已经使用的内存占比，即（1047543808-717537280）/1047543808\*100% 。available表示还可以使用的内存。

    info['disk_partitions'] = psutil.disk_partitions()
    # [sdiskpart(device='/dev/mapper/root', mountpoint='/', fstype='ext4', opts='rw,errors=remount-ro'), sdiskpart(device='/dev/sda1', mountpoint='/boot', fstype='ext2', opts='rw')]
    info['disk_usage'] = psutil.disk_usage('/')._asdict()
    # sdiskusage(total=42273669120, used=17241096192, free=22885195776, percent=40.8)

    info['disk_io_counters'] = psutil.disk_io_counters(
        perdisk=True)
    # {'vdb1': sdiskio(read_count=312, write_count=0, read_bytes=1238016, write_bytes=0, read_time=95, write_time=0), 'vda1': sdiskio(read_count=637878, write_count=77080257, read_bytes=16036557824, write_bytes=1628873314304, read_time=2307272, write_time=1777841879)}

    # 获取网卡的io情况
    info['net_io_counters'] = psutil.net_io_counters(pernic=True)
    # {'lo': snetio(bytes_sent=56524704027, bytes_recv=56524704027, packets_sent=33602236, packets_recv=33602236, errin=0, errout=0, dropin=0, dropout=0), 'eth0': snetio(bytes_sent=468966480940, bytes_recv=352622081327, packets_sent=914930488, packets_recv=744583332, errin=0, errout=0, dropin=0, dropout=0)}

    PID = os.getpid()
    p = psutil.Process(PID)

    pid_info = dict(
        name=p.name(),
        exe=p.exe(),
        cwd=p.cwd(),
        status=p.status(),
        create_time=p.create_time(),
        uids=p.uids(),
        gids=p.gids(),
        cpu_times=p.cpu_times(),
        memory_percent=p.memory_percent(),
        memory_info=p.memory_info(),
        connections=p.connections(),
        num_threads=p.num_threads(),
    )

    info['pid'] = pid_info

    return info
