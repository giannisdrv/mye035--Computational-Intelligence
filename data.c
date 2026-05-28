#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define Num_of_data_with_category 8000



FILE *fptr;

const char* calculate_category(float x1, float x2);

void generate_data_with_category(){

    fptr = fopen("training_data_with_category.txt", "w");
    float x1;
    float x2;

    for(int i=0; i<Num_of_data_with_category/2; i++){
        x1 = (float)rand() / RAND_MAX * 2.0;
        x2 = (float)rand() / RAND_MAX * 2.0;
        const char* category = calculate_category(x1,x2);
        fprintf(fptr, "%.3f \t %.3f \t %s\n",x1,x2,category);
    }
    fclose(fptr);

    fptr = fopen("testing_data_with_category.txt", "w");
    for(int i=0; i<Num_of_data_with_category/2; i++){
        x1 = (float)rand() / RAND_MAX * 2.0;
        x2 = (float)rand() / RAND_MAX * 2.0;
        const char* category = calculate_category(x1,x2);
        fprintf(fptr, "%.3f \t %.3f \t %s\n",x1,x2,category);
    }

    fclose(fptr);
}


const char* calculate_category(float x1, float x2){
    if ( (x1-0.5)*(x1-0.5) + (x2-0.5)*(x2-0.5) < 0.2 && x1>0.5 && x2>0.5 ){
        return "C1";
    }
    else if ( (x1-0.5)*(x1-0.5) + (x2-0.5)*(x2-0.5) < 0.2 && x1<0.5 && x2>0.5 ){
        return "C2";
    }
    else if ( (x1-0.5)*(x1-0.5) + (x2-0.5)*(x2-0.5) < 0.2 && x1>0.5 && x2<0.5 ){
        return "C2";
    }
    else if ( (x1-0.5)*(x1-0.5) + (x2-0.5)*(x2-0.5) < 0.2 && x1<0.5 && x2<0.5 ){
        return "C1";
    }
    else if ( (x1-1.5)*(x1-1.5) + (x2-0.5)*(x2-0.5) < 0.2 && x1>1.5 && x2>0.5 ){
        return "C1";
    }
    else if ( (x1-1.5)*(x1-1.5) + (x2-0.5)*(x2-0.5) < 0.2 && x1<1.5 && x2>0.5 ){
        return "C2";
    }
    else if ( (x1-1.5)*(x1-1.5) + (x2-0.5)*(x2-0.5) < 0.2 && x1>1.5 && x2<0.5 ){
        return "C2";
    }
    else if ( (x1-1.5)*(x1-1.5) + (x2-0.5)*(x2-0.5) < 0.2 && x1<1.5 && x2<0.5 ){
        return "C1";
    }
    else if ( (x1-0.5)*(x1-0.5) + (x2-1.5)*(x2-1.5) < 0.2 && x1>0.5 && x2>1.5 ){
        return "C1";
    }
    else if ( (x1-0.5)*(x1-0.5) + (x2-1.5)*(x2-1.5) < 0.2 && x1<0.5 && x2>1.5 ){
        return "C2";
    }
    else if ( (x1-0.5)*(x1-0.5) + (x2-1.5)*(x2-1.5) < 0.2 && x1>0.5 && x2<1.5 ){
        return "C2";
    }
    else if ( (x1-0.5)*(x1-0.5) + (x2-1.5)*(x2-1.5) < 0.2 && x1<0.5 && x2<1.5 ){
        return "C1";
    }
    else if ( (x1-1.5)*(x1-1.5) + (x2-1.5)*(x2-1.5) < 0.2 && x1>1.5 && x2>1.5 ){
        return "C1";
    }
    else if ( (x1-1.5)*(x1-1.5) + (x2-1.5)*(x2-1.5) < 0.2 && x1<1.5 && x2>1.5 ){
        return "C2";
    }
    else if ( (x1-1.5)*(x1-1.5) + (x2-1.5)*(x2-1.5) < 0.2 && x1>1.5 && x2<1.5 ){
        return "C2";
    }
    else if ( (x1-1.5)*(x1-1.5) + (x2-1.5)*(x2-1.5) < 0.2 && x1<1.5 && x2<1.5 ){
        return "C1";
    }
    else{
        if( (x1-1)*(x2-1)>0){
            return "C3";
        }else{
            return "C4";
        }
    }
}

void generate_data_without_category(){

    fptr = fopen("training_data_without_category.txt", "w");
    float x1;
    float x2;
    for(int i=0; i<150; i++){
        x1 = 0.75 + (float)rand() / RAND_MAX * 0.5; 
        x2 = 0.75 + (float)rand() / RAND_MAX * 0.5;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    for(int i=0; i<150; i++){
        x1 = (float)rand() / RAND_MAX * 0.5;
        x2 = (float)rand() / RAND_MAX * 0.5;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    for(int i=0; i<150; i++){
        x1 = (float)rand() / RAND_MAX * 0.5;
        x2 = 1.5 + (float)rand() / RAND_MAX * 0.5;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    for(int i=0; i<150; i++){
        x1 = 1.5 + (float)rand() / RAND_MAX * 0.5;
        x2 = (float)rand() / RAND_MAX * 0.5;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    for(int i=0; i<150; i++){
        x1 = 1.5 + (float)rand() / RAND_MAX * 0.5;
        x2 = 1.5 + (float)rand() / RAND_MAX * 0.5;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    for(int i=0; i<75; i++){
        x1 = 0.6 + (float)rand() / RAND_MAX * 0.2;
        x2 = (float)rand() / RAND_MAX * 0.4;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    for(int i=0; i<75; i++){
        x1 = 0.6 + (float)rand() / RAND_MAX * 0.2;
        x2 = 1.6 + (float)rand() / RAND_MAX * 0.4;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    for(int i=0; i<75; i++){
        x1 = 1.2 + (float)rand() / RAND_MAX * 0.2;
        x2 = (float)rand() / RAND_MAX * 0.4;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    for(int i=0; i<75; i++){
        x1 = 1.2 + (float)rand() / RAND_MAX * 0.2;
        x2 = 1.6 + (float)rand() / RAND_MAX * 0.4;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    for(int i=0; i<150; i++){
        x1 = (float)rand() / RAND_MAX * 2;
        x2 = (float)rand() / RAND_MAX * 2;
        fprintf(fptr, "%.3f \t %.3f\n",x1,x2);
    }
    fclose(fptr);
}


int main(void)
{
    srand(time(NULL));
    generate_data_with_category();
    generate_data_without_category();
    return 0;
}
