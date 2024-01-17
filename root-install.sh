#!/bin/bash
set -e
apt install libffi-dev libssl-dev git curl build-essential zlib1g-dev tk-dev libsqlite3-dev libbz2-dev liblzma-dev libreadline-dev libncurses-dev
dir=$(pwd)
cat < EOF /tmp/PrintServer.desktop
#!/usr/bin/env xdg-open

[Desktop Entry]
Name=Print Server
Encoding=UTF-8
Exec=bash -c "cd $dir && $dir/start.sh"
Type=Application
Categories=Development;
Terminal=true
EOF

desktop-file-install /tmp/PrintServer.desktop
