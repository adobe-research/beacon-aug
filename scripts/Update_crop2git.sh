#!/bin/bash
cd ../docs
find . -name '*' -exec sed -i -e 's/git.corp.adobe.com/github.com/g' {} \;


find . -type f -name "*-e" -delete

find . -type f -name ".!?????!**" -delete
