#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import stat
import mechanize
import shutil
import zipfile


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != \
            "--i-accept-the-What-If-Pre-Release-License-Agreement-and-please-take-my-soul":
        print("Sorry, this tool is only ment to be used by automated build systems.")
        print("Please get IACA 'the-regular-way' from " 
              "https://software.intel.com/protected-download/267266/157552")
        print("")
        print("However, if you happen to be an automated build system: "
              "there is something wrong with your configuration.")
        sys.exit(1)
        
    br = mechanize.Browser()
    br.open("https://software.intel.com/protected-download/267266/157552")
    br.select_form(nr=1)
    br['accept_license'] = ['1']
    br.submit()
    link = br.find_link(text_regex='iaca-lin64\.zip')
    filename, headers = br.retrieve(link.absolute_url)
    #shutil.move(filename, 'iaca-lin64.zip')
    #filename = 'iaca-lin64.zip'
    zfile = zipfile.ZipFile(filename)
    members = [n for n in zfile.namelist() if '/.' not in n and n.startswith('iaca-lin64/')]
    zfile.extractall(members=members)
    
    st = os.stat('iaca-lin64/bin/iaca')
    os.chmod('iaca-lin64/bin/iaca', st.st_mode | stat.S_IEXEC)
    st = os.stat('iaca-lin64/bin/iaca.sh')
    os.chmod('iaca-lin64/bin/iaca.sh', st.st_mode | stat.S_IEXEC)
    
    print("{:}/iaca-lin64/bin/".format(os.getcwd()))
    
    
