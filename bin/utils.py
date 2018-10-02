import datetime
import sys

def log(string):
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
