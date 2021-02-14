#! /bin/bash


python -c 'import sys; print (sys.real_prefix)' 2>/dev/null && INVENV=1 || INVENV=0

if [ ! $INVENV ]; then
    echo "please activate the virtual env"
    exit 1
fi

pyinstaller -F --key="$(cat password.txt)" --clean -y render.spec

cp dist/render ~/.config/jc-electron/bin/
# pyinstaller -F --key="test" render.py
