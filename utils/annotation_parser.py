def parse_annotation(label_list):

    parsed = {}

    for file in label_list:
        seq_name = file.split("/")[-1]
        seq_name = seq_name.split(".")[0]
        parsed[seq_name] = {"fname": [],
                            "p1": [],
                            "p2": [],
                            "p3": [],
                            "p4": []}
        with open(file, "r") as f:
            for line in f:
                split = line.split(" ")
                if split[1] == "-1":
                    continue
                parsed[seq_name]["fname"].append(split[0])
                parsed[seq_name]["p1"].append([split[1], split[2]])
                parsed[seq_name]["p2"].append([split[3], split[4]])
                parsed[seq_name]["p3"].append([split[5], split[6]])
                parsed[seq_name]["p4"].append([split[7], split[8]])

    return parsed


if __name__ == "__main__":
    pass

