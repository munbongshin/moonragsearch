import faiss
import numpy as np

class FAISSManager:
    def __init__(self):
        self.databases = {}

    def create_db(self, name, dimension):
        if name in self.databases:
            print(f"데이터베이스 '{name}'이(가) 이미 존재합니다.")
        else:
            index = faiss.IndexFlatL2(dimension)
            self.databases[name] = index
            print(f"데이터베이스 '{name}'이(가) 생성되었습니다.")

    def delete_db(self, name):
        if name in self.databases:
            del self.databases[name]
            print(f"데이터베이스 '{name}'이(가) 삭제되었습니다.")
        else:
            print(f"데이터베이스 '{name}'을(를) 찾을 수 없습니다.")

    def list_dbs(self):
        if self.databases:
            print("데이터베이스 목록:")
            for name in self.databases:
                print(f"- {name}")
        else:
            print("데이터베이스가 없습니다.")

    def add_vectors(self, name, vectors):
        if name in self.databases:
            self.databases[name].add(vectors)
            print(f"{len(vectors)}개의 벡터가 '{name}' 데이터베이스에 추가되었습니다.")
        else:
            print(f"데이터베이스 '{name}'을(를) 찾을 수 없습니다.")

    def search(self, name, query_vector, k=5):
        if name in self.databases:
            D, I = self.databases[name].search(np.array([query_vector]), k)
            print(f"'{name}' 데이터베이스에서 가장 가까운 {k}개의 벡터:")
            for i, (dist, idx) in enumerate(zip(D[0], I[0])):
                print(f"{i+1}. 인덱스: {idx}, 거리: {dist}")
        else:
            print(f"데이터베이스 '{name}'을(를) 찾을 수 없습니다.")

def main():
    manager = FAISSManager()

    while True:
        print("\n1. 데이터베이스 생성")
        print("2. 데이터베이스 삭제")
        print("3. 데이터베이스 목록")
        print("4. 벡터 추가")
        print("5. 벡터 검색")
        print("6. 종료")

        choice = input("선택하세요 (1-6): ")

        if choice == '1':
            name = input("데이터베이스 이름: ")
            dimension = int(input("벡터 차원: "))
            manager.create_db(name, dimension)
        elif choice == '2':
            name = input("삭제할 데이터베이스 이름: ")
            manager.delete_db(name)
        elif choice == '3':
            manager.list_dbs()
        elif choice == '4':
            name = input("벡터를 추가할 데이터베이스 이름: ")
            num_vectors = int(input("추가할 벡터 수: "))
            dimension = manager.databases[name].d
            vectors = np.random.rand(num_vectors, dimension).astype('float32')
            manager.add_vectors(name, vectors)
        elif choice == '5':
            name = input("검색할 데이터베이스 이름: ")
            if name in manager.databases:
                dimension = manager.databases[name].d
                query = np.random.rand(dimension).astype('float32')
                k = int(input("검색할 최근접 이웃 수: "))
                manager.search(name, query, k)
            else:
                print(f"데이터베이스 '{name}'을(를) 찾을 수 없습니다.")
        elif choice == '6':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()