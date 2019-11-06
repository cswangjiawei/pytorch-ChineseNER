if __name__ == '__main__':
    with open('msra.txt', 'r', encoding='utf-8') as f1:
        with open('msra_train.txt', 'w', encoding='utf-8') as f2:
            with open('msra_test.txt', 'w', encoding='utf-8') as f3:
                lines = f1.readlines()
                num = 0
                for line in lines:
                    if num < 50289:
                        f2.write(line)
                    else:
                        f3.write(line)
                    if len(line) < 3:
                        num += 1