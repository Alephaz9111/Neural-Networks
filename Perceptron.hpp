#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

#define PATTERN         4           //The Number Of The Patterns
#define NEURON_S        2           //The Number Of The S Layer's Neurons 
#define NEURON_A        2           //The Number Of The A Layer's Neurons 
#define NEURON_R        1           //The Number Of The R Layer's Neurons 
#define RATE            0.01        //Learning Late
#define UPDATEMAX       100000      //The Maximum Value Of Update (The Number Of Epoch)
#define TIMES           10          //The Parameter Of Success Late 

class Perceptron {
    public:
    Perceptron ();
    ~Perceptron ();
    double error ( double tg, int pattern );
    void weight_init ();
    void forward_propagation ( double** ip, double* tg, int pattern );
    void back_propagation ( double error_data, int pattern );
    void update_weight ( double** ip );
    protected:
    double change;
    double** w1 = NULL;
    double** w2 = NULL;
    double** hidden = NULL;
    double* output = NULL;
    double** d_output = NULL;
    double** d_hidden = NULL;
};

Perceptron::Perceptron () {

    w1 = new double* [NEURON_A+1];
    for ( int w = 0; w < NEURON_A+1; w++ ) {
        w1[w] = new double [NEURON_S+1];
    }

    w2 = new double* [NEURON_R+1];
    for ( int w = 0; w < NEURON_R+1; w++ ) {
        w2[w] = new double [NEURON_A+1];
    }

    hidden = new double* [PATTERN];
    for ( int p = 0; p < PATTERN; p++ ) {
        hidden[p] = new double [NEURON_A+1];
    }

    output = new double [PATTERN];

    d_output = new double* [PATTERN];
    for ( int p = 0; p < PATTERN; p++ ) {
        d_output[p] = new double [NEURON_R];
    }

    d_hidden = new double* [PATTERN];
    for ( int p = 0; p < PATTERN; p++ ) {
        d_hidden[p] = new double [NEURON_A];
    }

};

Perceptron::~Perceptron () {
    
    delete w1;
    delete w2;
    delete hidden;
    delete output;
    delete d_output;
    delete d_hidden;

}

double Perceptron::error ( double tg, int pattern ) {
    return tg - output[pattern];
}

void Perceptron::weight_init () {

    srand((unsigned)time(NULL));
    for ( int v = 0; v < NEURON_A+1; v++ ) {
        for ( int w = 0; w < NEURON_S+1; w++ ) {
            w1[v][w] = (double)(rand() % 2001) / 1000.0 - 1.0;
        }
    }

}

void Perceptron::forward_propagation ( double** ip, double* tg, int pattern ) {
    
    /* Input => Hidden */
    for ( int j = 0; j < NEURON_A; j++ ) {
        double net = 0;
        for ( int i = 0; i < NEURON_S+1; i++ ) {
            net += w1[j][i] * ip[pattern][i];
        }
        hidden[pattern][j] = 1/(1+exp(0-net));
    }
    hidden[pattern][NEURON_A] = 1;

    /* Hidden => Output */
    for ( int j = 0; j < NEURON_R; j++ ) {
        double net = 0;
        for ( int i = 0; i < NEURON_A+1; i++ ) {
            net += w2[j][i] * hidden[pattern][i];
        }
        output[pattern] = 1/(1+exp(0-net));
    }

}

void Perceptron::back_propagation ( double error_data, int pattern ) {
    
    /* Output Layer */
    for ( int i = 0; i < NEURON_R; i++ ) {
        d_output[pattern][i] = output[pattern] * ( 1 - output[pattern] ) * error_data;
    }

    /* Hidden Layer */
    for ( int j = 0; j < NEURON_A; j++ ){
        d_hidden[pattern][j] = 0;
        for ( int k = 0; k < NEURON_R; k++ ) {
            d_hidden[pattern][j] += hidden[pattern][j] * ( 1 - hidden[pattern][j] ) * (w2[k][j] * d_output[pattern][k]);
        }
    }

}

void Perceptron::update_weight ( double** ip ) {

    for ( int v = 0; v < NEURON_A; v++ ) {
        for ( int w = 0; w < NEURON_S+1; w++ ) {
            change = 0;
            for ( int p = 0; p < PATTERN; p++ ) {
                change += RATE * d_hidden[p][v] * ip[p][w];
            }
            w1[v][w] += change;
        }
    }

    for ( int u = 0; u < NEURON_R; u++ ) {
        for ( int v = 0; v < NEURON_A+1; v++ ) {
            change = 0;
            for ( int p = 0; p < PATTERN; p++ ) {
                change += RATE * d_output[p][u] * hidden[p][v];
            }
            w2[u][v] += change;
        }
    }

}