#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import stat
import zipfile
import re
from io import BytesIO

import requests


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != \
            "--i-accept-the-What-If-Pre-Release-License-Agreement-and-please-take-my-soul":
        print("Sorry, this tool is only ment to be used by automated build systems.")
        print("Please get IACA 'the-regular-way' from " 
              "https://software.intel.com/protected-download/267266/157552")
        print("")
        print("However, if you happen to be an automated build system: "
              "there is something wrong with your configuration.")
        sys.exit(1)
    
    assert sys.argv[2] in ['lin64', 'lin32', 'mac32', 'mac64']
    version = sys.argv[2]
    
    URL = "https://software.intel.com/protected-download/267266/157552"
        
    s = requests.Session()
    r = s.get(URL)
    response_data = {
        'accept_license': 1,
        'form_build_id': re.search(r'name="form_build_id" value="([^"]+)" />', r.text).group(1),
        'form_id': 'intel_licensed_dls_step_1'}
    r = s.post(URL, data=response_data)
    download_url = re.search(
        r'"(https://software.intel.com/[^"]*iaca-'+version+'[^"]*\.zip)"', r.text).group(1)
    r = s.get(download_url, stream=True)
    zfile = zipfile.ZipFile(BytesIO(r.content))
    members = [n 
               for n in zfile.namelist() 
               if '/.' not in n and n.startswith('iaca-{:}/'.format(version))]
    zfile.extractall(members=members)
    
    st = os.stat('iaca-{:}/bin/iaca'.format(version))
    os.chmod('iaca-{:}/bin/iaca'.format(version), st.st_mode | stat.S_IEXEC)
    st = os.stat('iaca-{:}/bin/iaca.sh'.format(version))
    os.chmod('iaca-{:}/bin/iaca.sh'.format(version), st.st_mode | stat.S_IEXEC)
    
    # Fixing iaca.sh
    iaca_sh = open('iaca-{:}/bin/iaca.sh'.format(version)).read()
    iaca_sh = iaca_sh.replace('realpath', 'readlink -f', 1)
    open('iaca-{:}/bin/iaca.sh'.format(version), 'w').write(iaca_sh)
    
    print("{:}/iaca-{:}/bin/".format(os.getcwd(), version))
    
    
