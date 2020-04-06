#include "Perceptron.hpp"
using namespace std;

int main () {

    int count, success;
    double err, change;
    bool flag = false;
    double** input = NULL;
    double* target = NULL;
    double* E = new double [PATTERN];
    Perceptron pct;

    input = new double* [PATTERN];
    for ( int p = 0; p < PATTERN; p++ ) {
        input[p] = new double [NEURON_S+1];
    }

    target = new double [PATTERN];

    /* ( x1, x2 ) = ( 0, 0 ) */
    input[0][0] = 0;
    input[0][1] = 0;
    /* ( x1, x2 ) = ( 0, 1 ) */
    input[1][0] = 0;
    input[1][1] = 1;
    /* ( x1, x2 ) = ( 1, 0 ) */
    input[2][0] = 1;
    input[2][1] = 0;
    /* ( x1, x2 ) = ( 1, 1 ) */
    input[3][0] = 1;
    input[3][1] = 1;

    /* x3 = 1 */
    for( int p = 0; p < PATTERN; p++ ) input[p][2] = 1;

    /* XOR */
    target[0] = 0;
    target[1] = 1;
    target[2] = 1;
    target[3] = 0;

    success = 0;
    for ( int t = 0; t < TIMES; t++ ) {

        pct.weight_init();

        count = 0;
        while ( count < UPDATEMAX && flag == false ) {

            for ( int p = 0; p < PATTERN; p++ ) {
                pct.forward_propagation( input, target, p );
                /* Error Between Teacher Data And Output */
                err = pct.error( target[p], p );
                pct.back_propagation( err, p );
                /* Each Pattarns Of Square Error */
                E[p] = 0.5 * pow( err, 2.0 );
            }

            pct.update_weight( input );

            double Esum = 0;
            for ( int p = 0; p < PATTERN; p++ ) {
                Esum += E[p];
            }
            
            std::cout << count << " | E : " << Esum << "\n";
            
            if ( Esum < 0.01 ) flag = true;
            count++;
            
        }
        
        if ( count != UPDATEMAX ) success++;

    }

    /* SUCCESS RATE */
    std::cout << "SUCCESS : " << (double)success/TIMES * 100 << "%" << std::endl;

    pct.~Perceptron();

    return 0;
}
