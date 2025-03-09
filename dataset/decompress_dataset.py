#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
from pathlib import Path
from sys import platform

Description = """
Decompression of archives created by the PPC_bench.py tool

Important note: the decompression of the archives can create 
directories containing a very large number of files, so large
indeed that command line globbing does not work (hint: to list
files in such directory use `/bin/ls -R` or `find` rather than 
simple `ls`; to delete one such directory use `rm -fr`)

To avoid having a working directory flooded by extraneous files that 
can not be easily deleted, the script force the destination directory
to be empty when decompression begins. 

Unfortunately the decompression of the tar archives cannot be done
in parallel since different tar processes writing to the same directory 
can cause race conditions see:
  https://www.gnu.org/software/tar/manual/html_section/Reliability.html
"""

# number of leading digits in multifile archive
MFAdigits = 9

# execute the command in split_cmd where the single components
# of the command line have been already parsed


def execute_command(split_cmd):
    # build string command line
    command = ""
    for w in split_cmd:
        if len(w.split()) > 1:
            command += f"'{w}' "
        else:
            command += f"{w} "
    if args.verbose:
        print('Executing:\n  ' + command)
    try:
        subprocess.run(split_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as ex:
        print("Error executing command line:")
        print("\t" + command)
        print("--- stderr ---")
        print(ex.stderr.decode("utf-8"))
        return False
    except Exception as ex:
        print("Error executing subprocess invocation on the command line:")
        print("\t" + command)
        print(ex)
        return False
    return True


# decompress one or more tar archive
def decompress_archives(infile):
    main = os.path.basename(infile)
    path = os.path.dirname(infile)
    archives = [main]
    # check if there are secondary files, ie main starts with ten 0s
    if main[:MFAdigits] == MFAdigits*"0":
        print("This appears to be a multifile archive")
        suffix = main[MFAdigits:]
        for i in range(1, 10**MFAdigits):
            prefix = (MFAdigits-len(str(i)))*"0"+str(i)
            if os.access(os.path.join(path, prefix+suffix), os.F_OK):
                archives.append(prefix+suffix)
            else:
                break
    # we now have the complete list of archives
    if args.verbose:
        print(f"Decompressing {len(archives)} archive(s)")
    tot_ok = 0
    for f in archives:
        # decompress archive
        ca = os.path.join(path, f)
        # build the command line to be executed
        # if mac gtar invece che tar
        tar = 'tar'

        if platform == "darwin":
            tar = 'gtar'

        command = [tar, '-xf', ca, '-I', args.c]
        ok = execute_command(command)
        if ok:
            tot_ok += 1
    if tot_ok == len(archives):
        print("All archives successfully extracted!")
    else:
        print(f"{len(archives)-tot_ok} extractions failed! Check error messages.")


# check comand line parameters and launch decompression
if __name__ == "__main__":
    # Instantiate the parser: options and arguments are in the global variable args
    parser = argparse.ArgumentParser(
        description=Description, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('main', metavar="main-archive", type=str,
                        help='main archive file name')
    parser.add_argument('--output', '-o', type=str,
                        help='output directory, must be new or empty')
    parser.add_argument('-c', metavar='compressor', type=str, default='zstd',
                        help='compressor used to create archive (def. zstd)')
    parser.add_argument('--dataset', action='store_true',
                        help='use special dataset decompression options')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose output')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force download in the output directory, even if it exists and is not empty.')

    args = parser.parse_args()

  # check input file
    if not os.path.isabs(args.main):
        print(f"Fatal: input file must be an absolute path: {args.main}")
        sys.exit(1)

    if not os.access(args.main, os.F_OK):
        print(f"Fatal: missing input file: {args.main}")
        sys.exit(1)
    if not os.access(args.main, os.R_OK):
        print(f"Fatal: Cannot read input file: {args.main}")
        sys.exit(1)

    # create output dir if necessary
    outdir = os.path.abspath(args.output)
    print(f"Output directory: {outdir}")
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception as e:
        print(f"Fatal: Cannot create output directory {outdir}\n", e)
        sys.exit(1)

    # check access
    if not os.access(outdir, os.W_OK):
        print(f"Fatal: Cannot write to output directory {outdir}")
        sys.exit(1)

    # make sure dir is empty
    path_outdir = Path(outdir)
    if next(path_outdir.iterdir(), None) != None and not args.force:
        print(f"Fatal: output directory {outdir} is not empty")
        sys.exit(1)

    # enter into output directory
    os.chdir(outdir)
    # check if using --dataset option
    if args.dataset:
        if args.c != 'zstd':
            print("Fatal: option --dataset incompatible with -c")
            sys.exit(1)
        args.c = 'zstd --long=30 --adapt -M1024MB'

    # decompression
    decompress_archives(args.main)
    sys.exit(0)
