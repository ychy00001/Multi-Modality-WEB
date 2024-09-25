import os

def generate_check_file(input_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".jpg"):
            with open(os.path.join(input_dir, file.replace(".jpg", "_check.txt")), "w") as f:
                f.write('{"check": true, "type": "抛洒物"}')
        if file.endswith(".png"):
            with open(os.path.join(input_dir, file.replace(".png", "_check.txt")), "w") as f:
                f.write('{"check": true, "type": "抛洒物"}')

if __name__ == '__main__':
    INPUT_DIR = "/Users/rain/company/telecom/code/Multi-Modality-WEB/script/InternVL/data/images"
    generate_check_file(input_dir=INPUT_DIR)