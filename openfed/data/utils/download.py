import os


def wget_https(url: str, output_dir: str):
    cmd = f"wget --no-check-certificate --no-proxy -P {output_dir} {url}"
    return os.system(cmd) == 0


def wget_google_driver_url(file_id: str, filename: str):
    cmd = f"""
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={file_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={file_id}" -O {filename} && rm -rf /tmp/cookies.txt
    """
    return os.system(cmd) == 0


def tar_xvf(tar_file: str, output_dir: str) -> bool:
    cmd = f"tar -xvf {tar_file} -C {output_dir}"
    return os.system(cmd) == 0
