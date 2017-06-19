#include "VSIDS.cuh"

__global__ void vsids_test()
{
    VSIDS vsids(10);
    vsids.block_var(7);

    Clause c;
    create_clause_on_dev(2, c);
    Lit l = mkLit(3, false);
    addLitToDev(l, c);
    Lit l2 = mkLit(8, true);
    addLitToDev(l2, c);


    for (int i = 0; i < 50; i++) {
        vsids.handle_clause(c);
    }

    int total = 0;
    int match = 0;

    for (int i = 0; i < 5; i++) {
        Lit lit = vsids.next_literal();
        printf("Lit = ");
        print_lit(lit);
        printf("\n");
        if (lit == l) {
            match++;
        }
        total++;
        vsids.print();
    }

    printf("There were %f%% of matches.\n", ((float)match * 100) / total);

    printf("Done.\n");
}

int main_test()
{
    vsids_test <<< 1, 1>>>();
    cudaDeviceReset();
}
