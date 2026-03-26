#!/bin/bash
set -e

MODULE_ID="spectra"
MOVE_HOST="${1:-${MOVE_HOST:-move.local}}"
DEST="/data/UserData/schwung/modules/audio_fx"

echo "Installing $MODULE_ID to $MOVE_HOST..."
scp -r "dist/$MODULE_ID" "root@$MOVE_HOST:$DEST/"
ssh root@$MOVE_HOST "chown -R ableton:users $DEST/$MODULE_ID"
echo "Done. Verify: ssh root@$MOVE_HOST 'ls -la $DEST/$MODULE_ID/'"
