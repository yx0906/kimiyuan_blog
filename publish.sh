#!/bin/bash

pelican content -s publishconf.py
aws s3 cp output/ s3://kimiyuan.com/ --recursive --exclude "*.DS_Store"
