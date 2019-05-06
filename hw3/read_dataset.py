from os.path import join,dirname
curr_dir = join(dirname(__file__), 'aclimdb\\train\\pos')
print(curr_dir)
dataset_path = join(curr_dir, "0_9.txt")
print(dataset_path)

with open(dataset_path, "r") as r:
    print(r.readlines())