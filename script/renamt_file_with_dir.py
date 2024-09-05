import os


def rename_file_with_dir(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                new_file_name = file.replace(".txt", "_check.txt")
                os.rename(os.path.join(root, file), os.path.join(root, new_file_name))


if __name__ == '__main__':
    INPUT_DIR = "/Users/rain/Downloads/fp_100_box_with_text_check"
    rename_file_with_dir(input_dir=INPUT_DIR)