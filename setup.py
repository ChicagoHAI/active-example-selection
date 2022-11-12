from setuptools import find_packages, setup

def read_requirements(filename: str):
    with open(filename) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req
            )
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            requirements.append(fix_url_dependencies(line))
    return requirements

setup(
    name="active-example-selection",
    version="1.0.0",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ChicagoHAI/active-example-selection",
    author="Yiming Zhang",
    author_email="yimingz0@uchicago.edu",
    license="MIT",
    packages=find_packages(
        where="src"
    ),
    package_dir={
        '': 'src'
    },
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.9",
)
