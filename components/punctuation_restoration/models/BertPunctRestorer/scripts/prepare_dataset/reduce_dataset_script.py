if __name__ == "__main__":
    # with open("./data/test.csv", encoding="utf-8") as f_in, \
    #      open("./data/test_reduced.csv", "w", encoding="utf-8") as f_out:
    with open("./data/train.csv", encoding="utf-8") as f_in, \
            open("./data/train_reduced.csv", "w", encoding="utf-8") as f_out:

        sample = []

        i = 0

        for line in f_in.readlines():
            f_out.write(line)

            if line == "\n":
                i += 1

            if i == 10000:
                break
