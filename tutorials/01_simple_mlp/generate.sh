python3 \
    ../../src/codegen/main.py \
    --input="." \
    --output="." \
    --precision="double"

rm -rf .vscode/ codegen/__pycache__ dump_model/__pycache__