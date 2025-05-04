#!/usr/bin/env python

from setuptools import setup, find_namespace_packages

PROJECT = 'nav2py_crowdnav_height_controller'

setup(name=PROJECT,
      version='1.0',
      description='nav2py_crowdnav_height_controller',
      author='Volodymyr Shcherbyna',
      author_email='dev@voshch.dev',
      packages=find_namespace_packages()
      )
