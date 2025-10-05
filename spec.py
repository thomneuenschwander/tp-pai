import sys

def process_enrollments(enrollments):
    if not (3 <= len(enrollments) <= 4):
        print("Erro: informe 3 ou 4 números de matrícula.")
        return
    if not all(isinstance(e, int) and 100000 <= e <= 999999 for e in enrollments):
        print("Erro: todas as matrículas devem ter 6 dígitos.")
        return

    total_sum = sum(enrollments)

    DS = total_sum % 3
    NR = total_sum % 2
    NC = total_sum % 2
    ND = total_sum % 4

    dataset_map = {0: "coronal", 1: "sagital", 2: "axial"}
    dataset = dataset_map[DS]

    regressor = "Linear" if NR == 0 else "XGBoost"

    classifier = "XGBoost" if NC == 0 else "SVM"

    deep_map = {0: "ResNet50", 1: "DenseNet", 2: "EfficientNet", 3: "MobileNet"}
    deep_model = deep_map[ND]

    print("Matrículas informadas:", enrollments)
    print("Soma das matrículas:", total_sum)
    print(f"DS = {DS} → Dataset atribuído: {dataset}")
    print(f"NR = {NR} → Regressor raso atribuído: {regressor}")
    print(f"NC = {NC} → Classificador raso atribuído: {classifier}")
    print(f"ND = {ND} → Modelo profundo atribuído: {deep_model}")


if __name__ == "__main__":
    try:
        args = [int(x) for x in sys.argv[1:]]
        process_enrollments(args)
    except ValueError:
        print("Erro: os números de matrícula devem possuir 6 dígitos.")
