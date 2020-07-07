import os
import sys
import unittest
import xmlrunner
import argparse
from tests import *

xml_file = os.environ.get('XML_RESULT', 'test_report.xml')

def main():
    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output=open(xml_file, 'w+'), outsuffix = ''),
        failfast=False,
        buffer=False,
        catchbreak=False)

if __name__ == '__main__':
    main()
