#!/usr/bin/env python2
from __future__ import print_function

import os
import sys
import stat
import mechanize
import zipfile


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
        
    br = mechanize.Browser()
    br.open("https://software.intel.com/protected-download/267266/157552")
    br.select_form(nr=1)
    br['accept_license'] = ['1']
    br.submit()
    link = br.find_link(text_regex='iaca-{:}\.zip'.format(version))
    filename, headers = br.retrieve(link.absolute_url)
    zfile = zipfile.ZipFile(filename)
    members = [n for n in zfile.namelist() if '/.' not in n and n.startswith('iaca-{:}/'.format(version))]
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
    
    
