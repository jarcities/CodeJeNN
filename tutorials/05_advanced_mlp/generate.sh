python3 \
    ../../src/api-core/main.py \
    --input="." \
    --output="." \
    --double \
    --custom_activation="custom_activation" \
    --debug \
    # --model_image

rm -rf .vscode/ api-core/__pycache__ dump_model/__pycache__