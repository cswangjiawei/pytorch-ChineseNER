from ChineseNER.model import load


if __name__ == '__main__':
    model = load()
    model.get_entity_from_file('F:\\1.txt', 'F:\\2.txt')

