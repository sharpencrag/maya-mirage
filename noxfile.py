import nox

# the Nox session assumes that you're using a virtualenv or environment with
# mayapy set up as the python interpreter.

# Creating a venv with mayapy requires creating a linked `python.exe` file in
# Maya's bin directory, pointing at mayapy.exe.  Cie la vie.

@nox.session(venv_params=["--system-site-packages"])
def tests(session):
    session.install(".")
    session.run("python", "-m", "unittest", "discover", "-s", "tests")
