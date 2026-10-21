def matmul_4x4(A: list[list[str]], B: list[list[str]]) -> None:
    for i in range(4):
        for j in range(4):
            print(*[f"{A[i][k]} * {B[k][j]}" for k in range(4)])
        

def matmul_8x8(A: list[list[str]], B: list[list[str]]) -> None:
    AA = [A[i][:4] for i in range(4)]
    AB = [A[i][4:] for i in range(4)]
    AC = [A[i+4][:4] for i in range(4)]
    AD = [A[i+4][4:] for i in range(4)]
    
    BA = [B[i][:4] for i in range(4)]
    BB = [B[i][4:] for i in range(4)]
    BC = [B[i+4][:4] for i in range(4)]
    BD = [B[i+4][4:] for i in range(4)]
    
    matmul_4x4(AA, BA)
    matmul_4x4(AA, BB)
    matmul_4x4(AB, BC)
    matmul_4x4(AB, BD)
    matmul_4x4(AC, BA)
    matmul_4x4(AC, BB)
    matmul_4x4(AD, BC)
    matmul_4x4(AD, BD)
    


def main():
    A = [["AA_0", "AA_1", "AA_2", "AA_3", "AB_0", "AB_1", "AB_2", "AB_3"],
         ["AA_4", "AA_5", "AA_6", "AA_7", "AB_4", "AB_5", "AB_6", "AB_7"],
         ["AA_8", "AA_9", "AA_A", "AA_B", "AB_8", "AB_9", "AB_A", "AB_B"],
         ["AA_C", "AA_D", "AA_E", "AA_F", "AB_C", "AB_D", "AB_E", "AB_F"],
         ["AC_0", "AC_1", "AC_2", "AC_3", "AD_0", "AD_1", "AD_2", "AD_3"],
         ["AC_4", "AC_5", "AC_6", "AC_7", "AD_4", "AD_5", "AD_6", "AD_7"],
         ["AC_8", "AC_9", "AC_A", "AC_B", "AD_8", "AD_9", "AD_A", "AD_B"],
         ["AC_C", "AC_D", "AC_E", "AC_F", "AD_C", "AD_D", "AD_E", "AD_F"]]

    B = [["BA_0", "BA_1", "BA_2", "BA_3", "BB_0", "BB_1", "BB_2", "BB_3"],
         ["BA_4", "BA_5", "BA_6", "BA_7", "BB_4", "BB_5", "BB_6", "BB_7"],
         ["BA_8", "BA_9", "BA_A", "BA_B", "BB_8", "BB_9", "BB_A", "BB_B"],
         ["BA_C", "BA_D", "BA_E", "BA_F", "BB_C", "BB_D", "BB_E", "BB_F"],
         ["BC_0", "BC_1", "BC_2", "BC_3", "BD_0", "BD_1", "BD_2", "BD_3"],
         ["BC_4", "BC_5", "BC_6", "BC_7", "BD_4", "BD_5", "BD_6", "BD_7"],
         ["BC_8", "BC_9", "BC_A", "BC_B", "BD_8", "BD_9", "BD_A", "BD_B"],
         ["BC_C", "BC_D", "BC_E", "BC_F", "BD_C", "BD_D", "BD_E", "BD_F"]]

    matmul_8x8(A, B)
    
if __name__ == "__main__":
    main()