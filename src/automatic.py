import main

pre_bcknd = "ideal"
bcknd = "ideal"
num_rec = None
d_size = 200
d_n = 5
vqr_n = 9

ml_type = ["vqc", "vqr", "qsvc", "qsvr"]

for ml in ml_type:
    for _ in range(5):
        if ml_type=="vqr":
            main.prep(ml, pre_bcknd, bcknd, d_size, vqr_n, num_rec)
        else:
            main.prep(ml, pre_bcknd, bcknd, d_size, d_n, num_rec)