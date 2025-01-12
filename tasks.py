import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "corrupted_mnist"
PYTHON_VERSION = "3.11"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

@task
def python(ctx):
    """ """
    ctx.run("which python" if os.name != "nt" else "where python")
    
@task
def git_add_all(ctx, message):
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")


@task
def git_add_all(ctx, message):
    """
    Add, commit, and push all changes in the current directory.
    
    Command line usage:
    $ invoke git-add-all "your commit message"
    
    Example:
    $ invoke git-add-all "updated all files"
    """
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")


@task
def git_single_file(ctx, filepath, message):
    """
    Add, commit, and push a single file to git.
    
    Command line usage:
    $ invoke git-single-file "path/to/file" "your commit message"
    
    Example:
    $ invoke git-single-file "src/main.py" "updated main function"
    """
    ctx.run(f"git add {filepath}")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")


@task
def git_folder(ctx, folder_path, message):
    """
    Add, commit, and push an entire folder to git.
    
    Command line usage:
    $ invoke git-folder "path/to/folder" "your commit message"
    
    Example:
    $ invoke git-folder "src/components" "updated components"
    """
    ctx.run(f"git add {folder_path}/*")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")


@task
def git_specific_files(ctx, files, message):
    """
    Add, commit, and push multiple specific files to git.
    Files should be provided as a list of file paths.
    
    Command line usage:
    $ invoke git-specific-files "['file1.py','file2.txt']" "your commit message"
    
    Example:
    $ invoke git-specific-files "['src/main.py','tests/test_main.py']" "updated main and tests"
    """
    # files should be a list of file paths
    for file in files:
        ctx.run(f"git add {file}")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")


@task
def git_branch(ctx, branch_name, message):
    """
    Checkout a branch, add all changes, commit, and push to that branch.
    
    Command line usage:
    $ invoke git-branch "branch-name" "your commit message"
    
    Example:
    $ invoke git-branch "feature/new-login" "implemented new login system"
    """
    ctx.run(f"git checkout {branch_name}")
    ctx.run("git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push origin {branch_name}")