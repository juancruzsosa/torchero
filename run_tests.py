import os
import sys
import unittest
import xmlrunner
import argparse
from tests import *

xml_file = os.environ.get('XML_RESULT', 'test_report.xml')

def main():
    with open(xml_file, 'wb+') as f:
        unittest.main(
            testRunner=xmlrunner.XMLTestRunner(output=f),
            failfast=False,
            buffer=False,
            catchbreak=False)

if __name__ == '__main__':
    main()
