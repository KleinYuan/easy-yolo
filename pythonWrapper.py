import subprocess

fp = "test.png"
command = "${PWD}/darknet detector test cfg/easy.data cfg/easy.cfg ${PWD}/backup/easy_final.weights %s" % fp
proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
output = proc.stdout.read()

# Now we take out the output into python variable from C
print output

# Do something about `output`
# .....
