import os

def generate_check_file(input_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".jpg"):
            with open(os.path.join(input_dir, file.replace(".jpg", "_check.txt")), "w") as f:
                f.write('{"check": true, "type": "\u6295\u6d88\u7269"}')

if __name__ == '__main__':
    INPUT_DIR = "/Users/rain/Downloads/fp_100_box_with_text_check"
    generate_check_file(input_dir=INPUT_DIR)