import os


def get_file_path(folder_name: str = None, file_name: str = None):
    dirpath = os.getcwd()
    filepath = f"{folder_name}/{file_name}"
    datapath = dirpath + "/" + filepath
    return datapath



