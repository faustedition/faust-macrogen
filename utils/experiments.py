from subprocess import run

with open("runs.lst") as f:
    for line in f:
        name, args = line[:-1].split('\t')
        cmd = f"macrogen --report-dir=../target/{name} --render-timeout=10 {args}"
        print(f"{name}: >>> {cmd}")
        run(cmd, shell=True)
        print(f"{name}: <<< {cmd}")
