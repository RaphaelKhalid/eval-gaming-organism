#!/bin/bash
# Builds bundle_v2.tgz from the committed working tree with LF line endings, then verifies it.
set -e
cd /path/to/evalgame
SP=/tmp/scratch
H=$(git rev-parse --short HEAD)
rm -rf $SP/stage/src $SP/stage/pod $SP/stage/wsl $SP/stage/COMMIT
cp -r src pod $SP/stage/; mkdir -p $SP/stage/wsl; cp wsl/gen.py wsl/monitor.py $SP/stage/wsl/
rm -rf $SP/stage/*/__pycache__
find $SP/stage \( -name "*.py" -o -name "*.sh" \) -exec sed -i 's/\r$//' {} +
echo $H > $SP/stage/COMMIT
(cd $SP/stage && tar czf /path/to/evalgame/bundle_v2.tgz COMMIT src pod wsl data)
rm -rf $SP/chk; mkdir $SP/chk; tar xzf bundle_v2.tgz -C $SP/chk
CR=$(python -c "import glob,sys;print(sum(b'\r' in open(f,'rb').read() for f in glob.glob(sys.argv[1]+'/**/*',recursive=True) if f.endswith(('.py','.sh'))))" $SP/chk)
echo "bundle commit=$(cat $SP/chk/COMMIT) files_with_CR=$CR size=$(stat -c %s bundle_v2.tgz) sha256=$(sha256sum bundle_v2.tgz | cut -c1-16)"
