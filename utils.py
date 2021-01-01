import csv


def append_to_csv(file_name, list, dir_name="./results"):
    with open("{}/{}".format(dir_name, file_name), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list)


def read_csv(file_name, dir_name="./results"):
    with open("{}/{}".format(dir_name, file_name), "r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        contents = [row for row in reader]
    return contents
