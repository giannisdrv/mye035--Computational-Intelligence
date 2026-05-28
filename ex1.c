#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#define d 2
#define K 4
#define N 4000
#define neurons_H1 20
#define neurons_H2 20
#define neurons_H3 20
#define activation_function_H1 "tanh"
#define activation_function_H2 "tanh"
#define activation_function_H3 "tanh"
#define max(a,b) ((a) > (b) ? (a) : (b))
#define threshold 0.1
#define learning_rate 0.001
#define epochs 800
#define L 20
#define batches N/L


float testing_data_with_category[N][d];
float training_data_with_category[N][d];
int training_categories[N][K];
int testing_categories[N][K];

float w1[d][neurons_H1]; //weights for layer1
float w2[neurons_H1][neurons_H2]; //weights for layer2
float w3[neurons_H2][neurons_H3]; //weights for layer3
float w4[neurons_H3][K]; //weights for output layer

float h1[neurons_H1]; //outputs of layer1
float h2[neurons_H2]; //outputs of layer2
float h3[neurons_H3]; //outputs of layer3
float output[K]; //outputs of output layer

float b1[neurons_H1]; //biases for layer1
float b2[neurons_H2]; //biases for layer2
float b3[neurons_H3]; //biases for layer3
float b4[K]; //biases for output layer

float dw1[d][neurons_H1]; //weight updates for layer1
float dw2[neurons_H1][neurons_H2]; //weight updates for layer2
float dw3[neurons_H2][neurons_H3]; //weight updates for layer
float dw4[neurons_H3][K]; //weight updates for output layer

float dh1[neurons_H1]; //delta for layer1
float dh2[neurons_H2]; //delta for layer2
float dh3[neurons_H3]; //delta for layer3
float doutput[K]; //delta for output layer

float db1[neurons_H1]; //bias updates for layer1
float db2[neurons_H2]; //bias updates for layer2
float db3[neurons_H3]; //bias updates for layer3
float db4[K]; //bias updates for output layer

float sum_db1[neurons_H1]; //sum of bias updates for layer1 in batch
float sum_db2[neurons_H2]; //sum of bias updates for layer2 in batch
float sum_db3[neurons_H3]; //sum of bias updates for layer3 in batch
float sum_db4[K]; //sum of bias updates for output layer in batch

float sum_dw1[d][neurons_H1]; //sum of weight updates for layer1 in batch
float sum_dw2[neurons_H1][neurons_H2]; //sum of weight updates for layer2 in batch
float sum_dw3[neurons_H2][neurons_H3]; //sum of weight updates for layer3 in batch
float sum_dw4[neurons_H3][K]; //sum of weight updates for output layer in batch

void softmax(float *input, int k);

void load_data_with_category(){
    FILE *fptr;
    char catergory[3];
    fptr = fopen("testing_data_with_category.txt", "r");
    for(int i=0; i<N; i++){
        fscanf(fptr, "%f %f %s", &testing_data_with_category[i][0], &testing_data_with_category[i][1], catergory);

        for(int j = 0; j < K; j++){
            testing_categories[i][j] = 0;
        }

        if (catergory[1] == '1'){
            testing_categories[i][0] = 1;
        }
        else if (catergory[1] == '2'){
            testing_categories[i][1] = 1;
        }
        else if (catergory[1] == '3'){
            testing_categories[i][2] = 1;
        }
        else if (catergory[1] == '4'){
            testing_categories[i][3] = 1;
        }
    }
    fclose(fptr);

    fptr = fopen("training_data_with_category.txt", "r");
    for(int i=0; i<N; i++){
        fscanf(fptr, "%f %f %s", &training_data_with_category[i][0], &training_data_with_category[i][1], catergory);
        
        for(int j = 0; j < K; j++){
            training_categories[i][j] = 0;
        }

        if (catergory[1] == '1'){
            training_categories[i][0] = 1;
        }
        else if (catergory[1] == '2'){
            training_categories[i][1] = 1;
        }
        else if (catergory[1] == '3'){
            training_categories[i][2] = 1;
        }
        else if (catergory[1] == '4'){
            training_categories[i][3] = 1;
        }
    }
    fclose(fptr);
}

float random_weights(){
    return -1.0 + ((float)rand() / RAND_MAX) * 2.0; 
}

void initialize_weights_and_biases(){
    srand(time(0));

    for(int i=0; i<d; i++){
        for(int j=0; j<neurons_H1; j++){
            w1[i][j] = random_weights();
        }
    }

    for(int i=0; i<neurons_H1; i++){
        for(int j=0; j<neurons_H2; j++){
            w2[i][j] = random_weights();
        }
    }

    for(int i=0; i<neurons_H2; i++){
        for(int j=0; j<neurons_H3; j++){
            w3[i][j] = random_weights();
        }
    }

    for(int i=0; i<neurons_H3; i++){
        for(int j=0; j<K; j++){
            w4[i][j] = random_weights();
        }
    }

    for(int i=0; i<neurons_H1; i++){
        b1[i] = random_weights();
    }

    for(int i=0; i<neurons_H2; i++){
        b2[i] = random_weights();
    }

    for(int i=0; i<neurons_H3; i++){
        b3[i] = random_weights();
    }

    for(int i=0; i<K; i++){
        b4[i] = random_weights();
    }
}

float logistic(float input){
    return 1.0 / (1.0 + exp(-input));
}

void forward_pass(float *x, int dim, float *y, int k){
    memset(h1, 0, sizeof(h1));
    memset(h2, 0, sizeof(h2)); 
    memset(h3, 0, sizeof(h3));
    memset(output, 0, sizeof(output));

    for (int j=0; j<neurons_H1; j++){
        float sum = b1[j];

        for (int i=0; i<d; i++){
            sum += x[i] * w1[i][j];
        }
        
        if (strcmp(activation_function_H1, "logistic") == 0){
            h1[j] = logistic(sum);
        }
        else if (strcmp(activation_function_H1, "tanh") == 0){
            h1[j] = tanh(sum);
        }
        else if (strcmp(activation_function_H1, "relu") == 0){
            h1[j] = max(0, sum);
        }
    }

    for (int j=0; j<neurons_H2; j++){
        float sum = b2[j];

        for (int i=0; i<neurons_H1; i++){
            sum += h1[i] * w2[i][j];
        }

        if (strcmp(activation_function_H2, "logistic") == 0){
            h2[j] = logistic(sum);
        }
        else if (strcmp(activation_function_H2, "tanh") == 0){
            h2[j] = tanh(sum);
        }
        else if (strcmp(activation_function_H2, "relu") == 0){
            h2[j] = max(0, sum);
        }
    }

    for (int j=0; j<neurons_H3; j++){
        float sum = b3[j];

        for (int i=0; i<neurons_H2; i++){
            sum += h2[i] * w3[i][j];
        }

        if (strcmp(activation_function_H3, "logistic") == 0){
            h3[j] = logistic(sum);
        }
        else if (strcmp(activation_function_H3, "tanh") == 0){
            h3[j] = tanh(sum);
        }
        else if (strcmp(activation_function_H3, "relu") == 0){
            h3[j] = max(0, sum);
        }
    }

    for (int j=0; j<K; j++){
        float sum = b4[j];

        for (int i=0; i<neurons_H3; i++){
            sum += h3[i] * w4[i][j];
        }
        output[j] = sum;
    }
    softmax(output, K);

    for (int j=0; j<K; j++){
        y[j] = output[j];
    }
}

void softmax(float *input, int k){
    float sum = 0.0;
    for (int i = 0; i < K; i++) {
        input[i] = exp(input[i]);
        sum += input[i];
    }

    for (int i = 0; i < K; i++) {
        input[i] /= sum;
    }
    
}

void derivative_relu(float *activation, float *delta, int size){
    for(int i=0; i<size; i++){
        delta[i] *= (activation[i] > 0) ? 1.0 : 0.0;  
    }
}

void derivative_tanh(float *activation, float *delta, int size){
    for(int i=0; i<size; i++){
        delta[i] *= (1.0 - activation[i] * activation[i]);  
    }
}

void derivative_logistic(float *activation, float *delta, int size){
    for(int i=0; i<size; i++){
        delta[i] *= activation[i] * (1.0 - activation[i]);  
    }
}


void backprop(float *x, int dim, float *t, int k){ 
    for (int i=0; i<K; i++){
        doutput[i] = output[i] - t[i];
    }

    for (int i=0; i<neurons_H3; i++){
        for (int j=0; j<K; j++){
            dw4[i][j] = h3[i] * doutput[j];
        }
    }

    for (int i=0; i<K; i++){
        db4[i] = doutput[i];
    }

    for (int i=0; i<neurons_H3; i++){
        dh3[i] = 0.0;
        for (int j=0; j<K; j++){
            dh3[i] += w4[i][j] * doutput[j];
        }
    }

    if (strcmp(activation_function_H3, "relu") == 0){
        derivative_relu(h3, dh3, neurons_H3);
    }
    else if (strcmp(activation_function_H3, "tanh") == 0){
        derivative_tanh(h3, dh3, neurons_H3);
    }
    else if (strcmp(activation_function_H3, "logistic") == 0){
        derivative_logistic(h3, dh3, neurons_H3);
    }

    for (int i=0; i<neurons_H2; i++){
        for (int j=0; j<neurons_H3; j++){
            dw3[i][j] = h2[i] * dh3[j];
        }
    }

    for (int i=0; i<neurons_H3; i++){
        db3[i] = dh3[i];
    }

    for (int i=0; i<neurons_H2; i++){
        dh2[i] = 0.0;
        for (int j=0; j<neurons_H3; j++){
            dh2[i] += w3[i][j] * dh3[j];
        }
    }

    if (strcmp(activation_function_H2, "relu") == 0){
        derivative_relu(h2, dh2, neurons_H2);
    }
    else if (strcmp(activation_function_H2, "tanh") == 0){
        derivative_tanh(h2, dh2, neurons_H2);
    }
    else if (strcmp(activation_function_H2, "logistic") == 0){
        derivative_logistic(h2, dh2, neurons_H2);
    }

    for (int i=0; i<neurons_H1; i++){
        for (int j=0; j<neurons_H2; j++){
            dw2[i][j] = h1[i] * dh2[j];
        }
    }

    for (int i=0; i<neurons_H2; i++){
        db2[i] = dh2[i];
    }

    for (int i=0; i<neurons_H1; i++){  
        dh1[i] = 0.0;
        for (int j=0; j<neurons_H2; j++){
            dh1[i] += w2[i][j] * dh2[j];
        }
    }

    if (strcmp(activation_function_H1, "relu") == 0){
        derivative_relu(h1, dh1, neurons_H1);
    }
    else if (strcmp(activation_function_H1, "tanh") == 0){
        derivative_tanh(h1, dh1, neurons_H1);
    }
    else if (strcmp(activation_function_H1, "logistic") == 0){
        derivative_logistic(h1, dh1, neurons_H1);
    }

    for (int i=0; i<d; i++){
        for (int j=0; j<neurons_H1; j++){
            dw1[i][j] = x[i] * dh1[j];
        }
    }

    for (int i=0; i<neurons_H1; i++){
        db1[i] = dh1[i];
    }
}

void calculate_sum_of_derivative_weights_and_biases(){
    for(int i=0; i<d; i++){
        for(int j=0; j<neurons_H1; j++){
            sum_dw1[i][j] += dw1[i][j];
        }
    }

    for(int i=0; i<neurons_H1; i++){
        for(int j=0; j<neurons_H2; j++){
            sum_dw2[i][j] += dw2[i][j];
        }
    }

    for(int i=0; i<neurons_H2; i++){
        for(int j=0; j<neurons_H3; j++){
            sum_dw3[i][j] += dw3[i][j];
        }
    }

    for(int i=0; i<neurons_H3; i++){
        for(int j=0; j<K; j++){
            sum_dw4[i][j] += dw4[i][j];
        }
    }

    for(int i=0; i<neurons_H1; i++){
        sum_db1[i] += db1[i];
    }

    for(int i=0; i<neurons_H2; i++){
        sum_db2[i] += db2[i];
    }

    for(int i=0; i<neurons_H3; i++){
        sum_db3[i] += db3[i];
    }

    for(int i=0; i<K; i++){
        sum_db4[i] += db4[i];
    }
}

void reset_sum_of_derivative_weights_and_biases(){
    memset(sum_dw1, 0, sizeof(sum_dw1));
    memset(sum_dw2, 0, sizeof(sum_dw2));
    memset(sum_dw3, 0, sizeof(sum_dw3));
    memset(sum_dw4, 0, sizeof(sum_dw4));
    memset(sum_db1, 0, sizeof(sum_db1));
    memset(sum_db2, 0, sizeof(sum_db2));
    memset(sum_db3, 0, sizeof(sum_db3));
    memset(sum_db4, 0, sizeof(sum_db4));
}   

void update_weights_and_biases(){
    float divide_samples = 1.0 / L;
    for(int i=0; i<d; i++){
        for(int j=0; j<neurons_H1; j++){
            w1[i][j] -= learning_rate * sum_dw1[i][j] * divide_samples;
        }
    }

    for(int i=0; i<neurons_H1; i++){
        for(int j=0; j<neurons_H2; j++){
            w2[i][j] -= learning_rate * sum_dw2[i][j] * divide_samples;
        }
    }

    for(int i=0; i<neurons_H2; i++){
        for(int j=0; j<neurons_H3; j++){
            w3[i][j] -= learning_rate * sum_dw3[i][j] * divide_samples;
        }
    }

    for(int i=0; i<neurons_H3; i++){
        for(int j=0; j<K; j++){
            w4[i][j] -= learning_rate * sum_dw4[i][j] * divide_samples;
        }
    }

    for(int i=0; i<neurons_H1; i++){
        b1[i] -= learning_rate * sum_db1[i] * divide_samples;
    }

    for(int i=0; i<neurons_H2; i++){
        b2[i] -= learning_rate * sum_db2[i] * divide_samples;
    }

    for(int i=0; i<neurons_H3; i++){
        b3[i] -= learning_rate * sum_db3[i] * divide_samples;
    }

    for(int i=0; i<K; i++){
        b4[i] -= learning_rate * sum_db4[i] * divide_samples;
    }
}

void train(){
    int epoch_counter = 0;
    float epoch_error = 0.0;
    float previous_epoch = 0.0;
    float current_epoch = 0.0;
    while (epoch_counter <800 || (epoch_counter >= 800 && epoch_error > threshold)){

        for (int i=0; i<batches; i++){
            reset_sum_of_derivative_weights_and_biases();

            int index = i * L;

            for (int j=index; j<index + L; j++){

                float t[K];
                float y[K];
                for(int k = 0; k < K; k++){
                    t[k] = (float)training_categories[j][k]; 
                }

                forward_pass(training_data_with_category[j], d, y, K);
                backprop(training_data_with_category[j], d, t, K);
                calculate_sum_of_derivative_weights_and_biases();
            }

            update_weights_and_biases();
        }

        float total_error = 0.0;
        
        
        for(int i = 0; i < N; i++){
            float y[K];  
            float t[K];  
            
            for(int k = 0; k < K; k++){
                t[k] = (float)training_categories[i][k];
            }
            
            forward_pass(training_data_with_category[i], d, y, K);
            
            for(int k = 0; k < K; k++){
                float diff = t[k] - y[k];
                total_error += 0.5 * diff * diff;
            }
        }
        
        float current_epoch_error = total_error / N;

        epoch_error = fabs(current_epoch_error - previous_epoch);
        epoch_counter ++;
        previous_epoch = current_epoch_error;
        printf("Epoch: %d. Error: %f. Error delta: %f\n", epoch_counter, current_epoch_error, epoch_error);
        
    }
}

void test_train(){
    int predicted_correct = 0;

    for(int i = 0; i < N; i++){
        float y[K];  
        
        forward_pass(testing_data_with_category[i], d, y, K);
        
        int predicted_category = 0;
        float max_prob = y[0];
        for(int k = 1; k < K; k++){
            if(y[k] > max_prob){
                max_prob = y[k];
                predicted_category = k;
            }
        }

        int actual_category = 0;
        for(int k = 0; k < K; k++){
            if(testing_categories[i][k] == 1){
                actual_category = k;
                break;
            }
        }

        if(predicted_category == actual_category){
            predicted_correct++;
            printf("%4d: +   pred=%d  actual=%d   \n", i, predicted_category+1, actual_category+1);
        }else{
            printf("%4d: -   pred=%d  actual=%d   \n", i, predicted_category+1, actual_category+1);
        }
    }

    float accuracy = (float)predicted_correct / N * 100.0;
    printf("Training Accuracy: %.2f%%\n", accuracy);
}


int main(void)
{
    load_data_with_category();
    initialize_weights_and_biases();
    train();
    test_train();
    return 0;
}
