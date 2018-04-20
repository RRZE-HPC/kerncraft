#!/usr/bin/env python3
import os
import sys
import stat
import zipfile
import re
from io import BytesIO
import tempfile
import shutil
import platform

import requests


class TemporaryDirectory:
    def __enter__(self):
        self.tempdir = tempfile.mkdtemp()
        return self.tempdir

    def __exit__(self, type_, value, traceback):
        shutil.rmtree(self.tempdir)


def get_os():
    os_map = {'Darwin': 'mac', 'Linux': 'lin64'}
    system = platform.system()
    assert system in os_map, "Unsupported operating system (platform.system() should return " \
                             "Linux or Darwin)."
    return os_map[system]


def serach_path():
    """Return potential locations of IACA installation."""
    operating_system = get_os()
    # 1st choice: in ~/.kerncraft/iaca-{}
    # 2nd choice: in package directory / iaca-{}
    return [os.path.expanduser("~/.kerncraft/iaca/{}/".format(operating_system)),
            os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/iaca/{}/'.format(
                operating_system)]


def find_iaca():
    """Return (hopefully) valid installation of IACA."""
    requires = ['iaca2.2', 'iaca2.3', 'iaca3.0']
    for path in serach_path():
        path += 'bin/'
        valid = True
        for r in requires:
            if not os.path.exists(path + r):
                valid = False
                break
        if valid:
            return path
    raise RuntimeError("No IACA installation found in {}. Run iaca_get command to fix this issue."
                       "".format(serach_path()))


def main():
    try:
        path = find_iaca()
        print('IACA already installed at', path)
        if '--force' in sys.argv:
            sys.argv.remove('--force')
        else:
            print('For forced installation add --force')
            sys.exit()
    except RuntimeError:
        pass
    if len(sys.argv) < 2 or sys.argv[1] != \
            "--I-accept-the-Intel-What-If-Pre-Release-License-Agreement-and-please-take-my-soul":
        print("Go to https://software.intel.com/protected-download/267266/157552 and read the"
              "Intel Pre-Release License Agreement.")
        print("")
        print("Add "
              "--I-accept-the-Intel-What-If-Pre-Release-License-Agreement-and-please-take-my-soul"
              " for installation of IACA.")
        sys.exit(1)

    if len(sys.argv) >= 3:
        assert sys.argv[2] in ['lin64', 'mac']
        operating_system = sys.argv[2]
    else:
        operating_system = get_os()

    # Locate and create IACA base directory, in reverse server order
    base_dir = None
    for path in reversed(serach_path()):
        print("Trying " + path + ": ", end='', file=sys.stderr)
        try:
            os.makedirs(path)
            base_dir = path
            break
        except PermissionError:
            # Continue trying with next location
            print("permission denied.", file=sys.stderr)
            continue
        except OSError:
            # Directory already exists
            print("already exists.", file=sys.stderr)
            continue
    if base_dir is None:
        print('Aborted.', file=sys.stderr)
        sys.exit(1)
    else:
        print("selected.", file=sys.stderr)

    URL = "https://software.intel.com/protected-download/267266/157552"

    s = requests.Session()
    r = s.get(URL)
    response_data = {
        'accept_license': 1,
        'form_build_id': re.search(r'name="form_build_id" value="([^"]+)" />', r.text).group(1),
        'form_id': 'intel_licensed_dls_step_1'}
    download_list = s.post(URL, data=response_data).text

    print("IACA v2.1 (for manual use - only version analyzing latency):", file=sys.stderr)
    if operating_system == 'mac':
        operating_system_temp = 'mac64'
    else:
        operating_system_temp = operating_system
    download_url = re.search(
        r'"(https://software.intel.com/[^"]*iaca-' + operating_system_temp + '\.zip)"',
        download_list).group(1)
    print("Downloading", download_url, "...", file=sys.stderr)
    r = s.get(download_url, stream=True)
    zfile = zipfile.ZipFile(BytesIO(r.content))
    members = [n
               for n in zfile.namelist()
               if '/.' not in n and n.startswith('iaca-{:}/'.format(operating_system_temp))]
    # Exctract to temp folder and copy to correct directory
    print("Extracting...", file=sys.stderr)
    with TemporaryDirectory() as tempdir:
        zfile.extractall(tempdir, members=members)
        shutil.copytree(tempdir + '/iaca-{}'.format(operating_system_temp), base_dir + 'v2.1')
    # Correct permissions of executables
    print("Correcting permissions of binary...")
    st = os.stat(base_dir + 'v2.1/bin/iaca')
    os.chmod(
        base_dir + 'v2.1/bin/iaca',
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    st = os.stat(base_dir + 'v2.1/bin/iaca.sh')
    os.chmod(
        base_dir + 'v2.1/bin/iaca.sh',
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    # Fix iaca.sh
    print("Fixing iaca.sh...", file=sys.stderr)
    iaca_sh = open(base_dir + 'v2.1/bin/iaca.sh').read()
    iaca_sh = iaca_sh.replace('realpath', 'readlink -f', 1)
    iaca_sh = iaca_sh.replace('mypath=`pwd`', 'mypath=`dirname $0`', 1)
    iaca_sh = iaca_sh.replace('path=$(cd "$(dirname "$0")"; pwd)',
                              'script=`readlink -f $0`\n\tpath=`dirname "$script"`', 1)
    open(base_dir + 'v2.1/bin/iaca.sh', 'w').write(iaca_sh)
    print("IACA v2.1 installed to", os.getcwd() + '/' + base_dir + 'v2.1', file=sys.stderr)

    print("IACA v2.2 (for NHM and WSM support):", file=sys.stderr)
    download_url = re.search(
        r'"(https://software.intel.com/[^"]*iaca-version-2.2-' + operating_system + '\.zip)"',
        download_list).group(1)
    print("Downloading", download_url, "...", file=sys.stderr)
    r = s.get(download_url, stream=True)
    zfile = zipfile.ZipFile(BytesIO(r.content))
    members = [n
               for n in zfile.namelist()
               if '/.' not in n and n.startswith('iaca-{:}/'.format(operating_system))]
    # Exctract to temp folder and copy to correct directory
    print("Extracting...", file=sys.stderr)
    with TemporaryDirectory() as tempdir:
        zfile.extractall(tempdir, members=members)
        shutil.copytree(tempdir + '/iaca-{}'.format(operating_system), base_dir + 'v2.2')
    # Correct permissions of executables
    print("Correcting permissions of binary...")
    st = os.stat(base_dir + 'v2.2/bin/iaca')
    os.chmod(
        base_dir + 'v2.2/bin/iaca',
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    st = os.stat(base_dir + 'v2.2/bin/iaca.sh')
    os.chmod(
        base_dir + 'v2.2/bin/iaca.sh',
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    # Fix iaca.sh
    print("Fixing iaca.sh...", file=sys.stderr)
    iaca_sh = open(base_dir + 'v2.2/bin/iaca.sh').read()
    iaca_sh = iaca_sh.replace('realpath', 'readlink -f', 1)
    iaca_sh = iaca_sh.replace('mypath=`pwd`', 'mypath=`dirname $0`', 1)
    iaca_sh = iaca_sh.replace('path=$(cd "$(dirname "$0")"; pwd)',
                              'script=`readlink -f $0`\n\tpath=`dirname "$script"`', 1)
    open(base_dir + 'v2.2/bin/iaca.sh', 'w').write(iaca_sh)
    print("IACA v2.2 installed to", os.getcwd() + '/' + base_dir + 'v2.2', file=sys.stderr)

    print("IACA v2.3 (for SNB and IVY support):", file=sys.stderr)
    download_url = re.search(
        r'"(https://software.intel.com/[^"]*iaca-version-2.3-' + operating_system + '\.zip)"',
        download_list).group(1)
    print("Downloading", download_url, "...", file=sys.stderr)
    r = s.get(download_url, stream=True)
    print("Reading zip file...", file=sys.stderr)
    zfile = zipfile.ZipFile(BytesIO(r.content))
    members = [n
               for n in zfile.namelist()
               if '/.' not in n and n.startswith('iaca-{:}/'.format(operating_system))]
    # Exctract to temp folder and copy to correct directory
    print("Extracting...", file=sys.stderr)
    with TemporaryDirectory() as tempdir:
        zfile.extractall(tempdir, members=members)
        shutil.copytree(tempdir + '/iaca-{}'.format(operating_system), base_dir + 'v2.3')
    # Correct permissions of executables
    print("Correcting permissions of binary...")
    st = os.stat(base_dir + 'v2.3/bin/iaca')
    os.chmod(
        base_dir + 'v2.3/bin/iaca',
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    st = os.stat(base_dir + 'v2.3/bin/iaca.sh')
    os.chmod(
        base_dir + 'v2.3/bin/iaca.sh',
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    # Fix iaca.sh
    print("Fixing iaca.sh...", file=sys.stderr)
    iaca_sh = open(base_dir + 'v2.3/bin/iaca.sh').read()
    iaca_sh = iaca_sh.replace('realpath', 'readlink -f', 1)
    iaca_sh = iaca_sh.replace('mypath=`pwd`', 'mypath=`dirname $0`', 1)
    iaca_sh = iaca_sh.replace('path=$(cd "$(dirname "$0")"; pwd)',
                              'script=`readlink -f $0`\n\tpath=`dirname "$script"`', 1)
    open(base_dir + 'v2.3/bin/iaca.sh', 'w').write(iaca_sh)
    print("IACA v2.3 installed to", os.getcwd() + '/' + base_dir + 'v2.3', file=sys.stderr)

    print("IACA v3.0 (for HSW, BDW, SKL and SKX support):", file=sys.stderr)
    download_url = re.search(
        r'"(https://software.intel.com/[^"]*iaca-version-v3.0-' + operating_system + '\.zip)"',
        download_list).group(1)
    print("Downloading...", download_url, "...", file=sys.stderr)
    r = s.get(download_url, stream=True)
    print("Reading zip file...", file=sys.stderr)
    zfile = zipfile.ZipFile(BytesIO(r.content))
    members = [n
               for n in zfile.namelist()
               if '/.' not in n and n.startswith('iaca-{:}/'.format(operating_system))]
    # Exctract to temp folder and copy to correct directory
    print("Extracting...", file=sys.stderr)
    with TemporaryDirectory() as tempdir:
        zfile.extractall(tempdir, members=members)
        shutil.copytree(tempdir + '/iaca-{}'.format(operating_system), base_dir + 'v3.0')

    print("Correcting permissions of binary...", file=sys.stderr)
    st = os.stat(base_dir + 'v3.0/iaca')
    os.chmod(
        base_dir + 'v3.0/iaca',
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    print("IACA v3.0 installed to", os.getcwd() + '/' + base_dir + 'v3.0', file=sys.stderr)

    # Create unified bin directory to access both operating_systems
    os.mkdir(base_dir + 'bin')
    os.symlink('../v2.1/bin/iaca.sh', base_dir + 'bin/iaca2.1')
    os.symlink('../v2.2/bin/iaca.sh', base_dir + 'bin/iaca2.2')
    os.symlink('../v2.3/bin/iaca.sh', base_dir + 'bin/iaca2.3')
    os.symlink('../v3.0/iaca', base_dir + 'bin/iaca3.0')
    print('export PATH=' + base_dir + 'bin/:$PATH')


if __name__ == '__main__':
    main()
