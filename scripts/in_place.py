import argparse

parser = argparse.ArgumentParser(description='Replace string in files')
parser.add_argument('-f', '--filename')
parser.add_argument('-o', '--old-string')
parser.add_argument('-n', '--new-string')

args = parser.parse_args()


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print '"{old_string}" not found in {filename}.'.format(**locals())
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print 'Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals())
        s = s.replace(old_string, new_string)
        f.write(s)

inplace_change(args.filename, args.old_string, args.new_string)
