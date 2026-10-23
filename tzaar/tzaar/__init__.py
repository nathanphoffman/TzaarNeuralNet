import os

# numpy's matrix maths starts a thread per core even for the tiny matrices used here, and on a many-core
# machine that overhead makes it 10 to 25 times slower. This has to be set before numpy is first imported.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
