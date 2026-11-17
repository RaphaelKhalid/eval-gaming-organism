#!/bin/bash
# Official runpodctl v3.1.1 (github.com/runpod/runpodctl), linux-amd64, checksum-verified.
set -e
mkdir -p ~/bin/rpdl && cd ~/bin/rpdl
B=https://github.com/runpod/runpodctl/releases/download/v3.1.1
curl -sL -o runpodctl-linux-amd64 $B/runpodctl-linux-amd64
curl -sL -o checksums.txt $B/checksums_3.1.1_sha256.txt
grep " runpodctl-linux-amd64$" checksums.txt | sha256sum -c -
install -m 755 runpodctl-linux-amd64 ~/bin/runpodctl
~/bin/runpodctl --version 2>&1 | head -1 || ~/bin/runpodctl version 2>&1 | head -1
