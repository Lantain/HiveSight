def generate_labels_file(labels, out_file):
    print(f"Saving labels: {labels}")
    if len(labels) == 0:
        return
    with open(out_file, 'w', encoding='UTF8') as f:
        i = 1
        for label in labels:
            str = f"item {{\n\tid: {i}\n\tname: '{label}'\n}}\n"
            f.writelines(str)
            i+=1
        