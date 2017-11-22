#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import stat
import zipfile
import re
from io import BytesIO
import tempfile
import shutil

import requests


class TemporaryDirectory:
    def __enter__(self):
        self.tempdir = tempfile.mkdtemp()
        return self.tempdir

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.tempdir)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != \
            "--I-accept-the-Intel-What-If-Pre-Release-License-Agreement-and-please-take-my-soul":
        print("Go to https://software.intel.com/protected-download/267266/157552 and accept the"
              "Intel Pre-Release License Agreement.")
        print("")
        print("Usage:", sys.argv[0],
              "--I-accept-the-Intel-What-If-Pre-Release-License-Agreement-and-please-take-my-soul",
              "{lin64, mac}")
        sys.exit(1)

    assert sys.argv[2] in ['lin64', 'mac']
    version = sys.argv[2]

    # Create IACA base directory
    base_dir = 'iaca-{}/'.format(version)
    try:
        os.mkdir(base_dir)
    except OSError:
        # Directory already exists
        print(base_dir, "already exists. Aborted.")
        sys.exit(1)

    URL = "https://software.intel.com/protected-download/267266/157552"

    s = requests.Session()
    r = s.get(URL)
    response_data = {
        'accept_license': 1,
        'form_build_id': re.search(r'name="form_build_id" value="([^"]+)" />', r.text).group(1),
        'form_id': 'intel_licensed_dls_step_1'}
    donwload_list = s.post(URL, data=response_data).text

    print("IACA v2.3 (for SNB and IVY support)", file=sys.stderr)
    download_url = re.search(
        r'"(https://software.intel.com/[^"]*iaca-version-2.3-'+version+'\.zip)"',
        donwload_list).group(1)
    print("Downloading", download_url, file=sys.stderr)
    r = s.get(download_url, stream=True)
    zfile = zipfile.ZipFile(BytesIO(r.content))
    members = [n
               for n in zfile.namelist()
               if '/.' not in n and n.startswith('iaca-{:}/'.format(version))]
    # Exctract to temp folder and copy to correct directory
    print("Extracting: {}".format(members), file=sys.stderr)
    with TemporaryDirectory() as tempdir:
        zfile.extractall(tempdir, members=members)
        shutil.copytree(tempdir+'/iaca-mac', base_dir+'v2.3')
    # Correct permissions of executables
    print("Correcting permissions of binary")
    st = os.stat(base_dir+'v2.3/bin/iaca')
    os.chmod(
        base_dir+'v2.3/bin/iaca',
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    st = os.stat(base_dir+'v2.3/bin/iaca.sh')
    os.chmod(
        base_dir+'v2.3/bin/iaca.sh',
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    # Fix iaca.sh
    print("Fixing iaca.sh", file=sys.stderr)
    iaca_sh = open(base_dir+'v2.3/bin/iaca.sh').read()
    iaca_sh = iaca_sh.replace('realpath', 'readlink -f', 1)
    iaca_sh = iaca_sh.replace('mypath=`pwd`', 'mypath=`dirname $0`', 1)
    iaca_sh = iaca_sh.replace('path=$(cd "$(dirname "$0")"; pwd)',
                              'script=`readlink -f $0`\n\tpath=`dirname "$script"`', 1)
    open(base_dir+'v2.3/bin/iaca.sh', 'w').write(iaca_sh)
    print("IACA v2.3 installed to", os.getcwd()+'/'+base_dir+'v2.3', file=sys.stderr)

    print("IACA v3.0 (for HSW, BDW, SKL and SKX support)", file=sys.stderr)
    download_url = re.search(
        r'"(https://software.intel.com/[^"]*iaca-version-v3.0-'+version+'\.zip)"',
        donwload_list).group(1)
    print("Downloading", download_url, file=sys.stderr)
    r = s.get(download_url, stream=True)
    print("Reading zip file", file=sys.stderr)
    zfile = zipfile.ZipFile(BytesIO(r.content))
    members = [n
               for n in zfile.namelist()
               if '/.' not in n and n.startswith('iaca-{:}/'.format(version))]
    # Exctract to temp folder and copy to correct directory
    print("Extracting: {}".format(members), file=sys.stderr)
    with TemporaryDirectory() as tempdir:
        zfile.extractall(tempdir, members=members)
        shutil.copytree(tempdir+'/iaca-mac', base_dir+'v3.0')

    print("Correcting permissions of binary", file=sys.stderr)
    st = os.stat(base_dir+'v3.0/iaca')
    os.chmod(
        base_dir+'v3.0/iaca'.format(version),
        st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    print("IACA v3.0 installed to", os.getcwd()+'/'+base_dir+'v3.0', file=sys.stderr)

    # Create unified bin directory to access both versions
    os.mkdir(base_dir+'bin')
    os.symlink('../v2.3/bin/iaca.sh', base_dir+'bin/iaca2.3')
    os.symlink('../v3.0/iaca', base_dir+'bin/iaca3.0')
    print('export PATH='+base_dir+'bin/:$PATH')
