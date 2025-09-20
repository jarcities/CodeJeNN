python3 \
    ../../src/codegen/main.py \
    --input="." \
    --output="." \
    --double \
    --debug

rm -rf .vscode/ codegen/__pycache__ dump_model/__pycache__