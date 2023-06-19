#bin/bash
echo "ChatMGL: There is a known issue in some environments, where trl does not work from requirements."
echo "If you get an error, please install trl manually using pip install trl"

cd src
python3 gen_script_chatMGL.py --model_path ../models/chatMGL/ --input_questions_path ../prompts.json --output_filename ../answers_chatMGL.json --generation_tokens 100
cd ..