from typing import List


def deduce_test_files(filepaths: List[str]) -> List[str]:
    test_files = list()
    for filepath in filepaths:
        extension_dot = filepath.rfind('.')
        test_file = filepath[:extension_dot] + '_tst' + filepath[extension_dot:]
        test_files.append(test_file)
    return test_files


def show_files_list(split_name: str, ground: str, preds: List[str]):
    print(f'{split_name.capitalize()} ground: {ground}')
    print('Prediction files:')
    for filename in preds:
        print(f'\t- {filename}')
    print()
