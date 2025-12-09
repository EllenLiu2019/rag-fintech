import os

PROJECT_ROOT_DIR = os.getenv("RAG_PROJECT_ROOT_DIR") or os.getenv("RAG_DEPLOY_ROOT_DIR")


def get_project_root_dir(*args):
    global PROJECT_ROOT_DIR
    if PROJECT_ROOT_DIR is None:
        PROJECT_ROOT_DIR = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
            )
        )

    if args:
        return os.path.join(PROJECT_ROOT_DIR, *args)
    return PROJECT_ROOT_DIR
