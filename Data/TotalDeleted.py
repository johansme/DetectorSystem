def calculate_deleted():
    with open('DeletedPerBatch.txt', 'r') as f:
        deleted = 0
        for line in f.readlines():
            if line != '':
                deleted += int(line)
    return deleted


print(calculate_deleted())
