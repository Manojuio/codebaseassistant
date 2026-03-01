from pathlib import Path

from rag.code.clone import clone_repo
from rag.code.ingest import ingest_code
from rag.pdf.ingest_pdf import ingest_pdf
from rag.core.vectorstore import build_vectorstore, load_vectorstore



SESSION_NAME = "active_session"


def get_or_create_store():
    store_path = Path("vectorstores/repo") / SESSION_NAME

    if store_path.exists():
        return load_vectorstore("repo", SESSION_NAME)

    return None


def main():
    print("=== Multi-Source Code Assistant ===")

    vectorstore = None
    llm = GroqLLM()

    while True:
        print("\nOptions:")
        print("1. Upload GitHub Repo")
        print("2. Upload PDF")
        print("3. Ask Question")
        print("4. Exit")

        choice = input("Select option: ").strip()

        # -----------------------------
        # 1️⃣ Upload Repo
        # -----------------------------
        if choice == "1":
            url = input("Enter GitHub URL: ").strip()

            print("Cloning repository...")
            repo_path = clone_repo(url)

            print("Ingesting repository...")
            docs = ingest_code(repo_path)

            repo_name = Path(repo_path).name

            vectorstore = build_vectorstore(
                documents=docs,
                store_type="repo",
                name=SESSION_NAME,
                force=False
            )

            print("Repository indexed successfully.")

        # -----------------------------
        # 2️⃣ Upload PDF
        # -----------------------------
        elif choice == "2":
            path = input("Enter PDF file path: ").strip()

            if not Path(path).exists():
                print("Invalid file path.")
                continue

            print("Processing PDF...")
            docs = ingest_pdf(path)

            vectorstore = build_vectorstore(
                documents=docs,
                store_type="repo",  # same session
                name=SESSION_NAME,
                force=False
            )

            print("PDF indexed successfully.")

        # -----------------------------
        # 3️⃣ Ask Question
        # -----------------------------
        elif choice == "3":
            if not vectorstore:
                print("No data indexed yet.")
                continue

            query = input("Ask your question: ").strip()

            print("Searching...")

            docs = vectorstore.similarity_search(query, k=5)

            context = "\n\n".join(
                f"[{d.metadata.get('source_type')} - {d.metadata.get('source_name')}]\n{d.page_content}"
                for d in docs
            )

            answer = llm.generate(context, query)

            print("\nAnswer:\n")
            print(answer)

        # -----------------------------
        # 4️⃣ Exit
        # -----------------------------
        elif choice == "4":
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()