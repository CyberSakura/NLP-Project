# -*- coding: utf-8 -*-

"""Top-level package for nlp."""

__author__ = """A Student"""
__email__ = 'student@example.com'
__version__ = '0.1.0'

# -*- coding: utf-8 -*-
import configparser
import os

# ~/.nlp/nlp.cfg will contain configuration information for the project,
# such as where data will be downloaded from.
# here is an example.
def write_default_config(path):
	w = open(path, 'wt')
	w.write('[data]\n')
	w.write('data_dir = %s%s%s%s%s\n' % (os.path.dirname(os.path.dirname(__file__)), os.path.sep, 'nlp', os.path.sep, 'data'))
	w.close()

# Find NLP_HOME path
if 'NLP_HOME' in os.environ:
    nlp_path = os.environ['NLP_HOME']
else:
    # Use USERPROFILE on Windows, HOME on Unix
    home = os.environ.get('USERPROFILE', os.environ.get('HOME', ''))
    nlp_path = os.path.join(home, '.nlp')

# Make nlp directory if not present
try:
    os.makedirs(nlp_path)
except:
    pass

# main config file.
config_path = os.path.join(nlp_path, 'nlp.cfg')
# classifier
clf_path = os.path.join(nlp_path, 'clf.pkl')

# write default config if not present.
if not os.path.isfile(config_path):
    write_default_config(config_path)

# config variable now accessible throughout project.
config = configparser.RawConfigParser()
config.read(config_path)