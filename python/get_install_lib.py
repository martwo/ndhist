from distutils import sysconfig
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option("", "--prefix", action="store", type="string", dest="PREFIX", metavar="PREFIX")
(options, args) = parser.parse_args()

install_lib_dir = os.path.join(options.PREFIX, sysconfig.get_python_lib()[len(sysconfig.PREFIX)+1:])
print(install_lib_dir)
