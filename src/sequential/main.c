#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_SAMPLES 512
#define N_FEATURES 784
#define THRESHOLD 0.5

double weights[N_FEATURES];  // array pesi
double bias = 0.0;

void init_model() {
    srand(1234);
    for(int i=0;i<N_FEATURES;i++){
        // piccoli numeri in [-0.01, +0.01]
        weights[i] = ((rand()/(double)RAND_MAX)-0.5)*0.02;
    }
    bias = 0.0;
}

static inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void load_csv(const char *filename, double data[N_SAMPLES][N_FEATURES]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Errore apertura file dati");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N_SAMPLES; i++) {
        for (int j = 0; j < N_FEATURES; j++) {
            if (fscanf(file, "%lf,", &data[i][j]) != 1) {
                fprintf(stderr, "Errore lettura dati in riga %d, colonna %d\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

void load_labels(const char *filename, int labels[N_SAMPLES]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Errore apertura file etichette");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N_SAMPLES; i++) {
        if (fscanf(file, "%d", &labels[i]) != 1) {
            fprintf(stderr, "Errore lettura etichetta in riga %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

int predict(double features[N_FEATURES]) {
    double sum = 0.0;
    for (int i = 0; i < N_FEATURES; i++) {
        sum += weights[i] * features[i];
    }
    return (sum > THRESHOLD) ? 1 : 0;
}

int main() {
    init_model();
    double X[N_SAMPLES][N_FEATURES];
    int y[N_SAMPLES];
    int correct = 0;

    // inizializza i pesi con maggiore importanza al centro (gaussiano semplice)
int width = 28;
int height = 28;
int center_i = height / 2;
int center_j = width / 2;

for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
        int idx = i * width + j;
        double dist = (i - center_i) * (i - center_i) + (j - center_j) * (j - center_j);
        weights[idx] = 1.0 / (1.0 + dist);  // più vicino al centro, più peso
    }
}


    printf("Caricamento dati...\n");
    load_csv("X_train.csv", X);
    load_labels("y_train.csv", y);
    printf("Dati caricati con successo.\n");

    // misura il tempo di esecuzione
    clock_t start = clock();

    for (int i = 0; i < N_SAMPLES; i++) {
        int pred = predict(X[i]);
        if (pred == y[i]) {
            correct++;
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Accuratezza: %.2f%%\n", (100.0 * correct / N_SAMPLES));
    printf("Tempo di esecuzione: %.6f secondi\n", elapsed);

    return 0;
}
