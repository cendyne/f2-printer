#!/bin/bash
set -e

export PATH="/usr/sbin/":$PATH

apt install libffi-dev libssl-dev git curl build-essential zlib1g-dev tk-dev libsqlite3-dev libbz2-dev liblzma-dev libreadline-dev libncurses-dev libcupsimage2
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
dir=$(pwd)
user=$(users)
cat << EOF > /tmp/PrintServer.desktop
#!/usr/bin/env xdg-open

[Desktop Entry]
Name=Print Server
Encoding=UTF-8
Exec=bash -c "cd '$dir' && '$dir'/start.sh"
Type=Application
Categories=Development;
Terminal=true
EOF

cat << EOF > /etc/sudoers.d/printing
%lpadmin ALL= NOPASSWD: /usr/bin/systemctl restart cups.service
EOF

sed -i 's/#HandleLidSwitch=ignore/HandleLidSwitch=ignore/' /etc/systemd/logind.conf

usermod -a -G lpadmin "$user"

desktop-file-install /tmp/PrintServer.desktop
