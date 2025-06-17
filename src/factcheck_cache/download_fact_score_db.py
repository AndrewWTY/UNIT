import os
import subprocess


_id = "1mekls6OGOKLmt7gYtHs0WGf5oTamTNat"
dest = "wiki_db/enwiki-20230401.db"

if "/" in dest:
    dest_dir = "/".join(dest.split("/")[:-1])
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
else:
    dest_dir = "."

command = """wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=%s" -O %s && rm -rf cookies.txt""" % (_id, _id, dest)
ret_code = subprocess.run([command], shell=True)

if ret_code.returncode != 0:
    print("Download {} ... [Failed]".format(dest))
else:
    print("Download {} ... [Success]".format(dest))

