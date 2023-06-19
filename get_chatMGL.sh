#bin/bash
CHATMGL_LOCATION=https://drive.google.com/drive/folders/1Klcx6gJHiJIj-BS6vB2-OM9Gx49qyxPd
OUTPUT_DIRECTORY=./models/chatMGL

gdown --folder $CHATMGL_LOCATION -O $OUTPUT_DIRECTORY

echo "Downloaded ChatMGL to $OUTPUT_DIRECTORY"