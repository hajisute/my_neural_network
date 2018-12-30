#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
//#include <jpeglib.h>
//num of units
#define NUM_INPUT_DIM 3 //2 + 1
#define NUM_OUTPUT_DIM 1
#define NUM_HIDDEN 100
#define NUM_TRAIN_DATA 4
#define NUM_TEST_DATA 4

double sigmoid(double x) 
{
    return 1/(1+exp(-x));
}

//derivative of sigmoid function
double d_sigmoid(double x) 
{
    double a = 0.1;
    return a*x*(1-x);
}

//train data
double train_x[NUM_TRAIN_DATA][NUM_INPUT_DIM] = {{0, 0, -1},{0, 1, -1},{1, 0, -1},{1, 1, -1}};
double train_d[NUM_TRAIN_DATA][NUM_OUTPUT_DIM] = {{0}, {1}, {1}, {0}};
//test data
double test_x[NUM_TRAIN_DATA][NUM_INPUT_DIM] = {{0, 0, -1},{0, 1, -1},{1, 0, -1},{1, 1, -1}};
double test_d[NUM_TRAIN_DATA][NUM_OUTPUT_DIM] = {{0}, {1}, {1}, {0}};

//ニューラルネット
double w[NUM_HIDDEN][NUM_INPUT_DIM][NUM_INPUT_DIM];//エッジの重み,w[層][from][to]

double x[NUM_HIDDEN][NUM_INPUT_DIM];//隠れ層の出力
double u[NUM_HIDDEN][NUM_INPUT_DIM];//隠れ層の入力
double train_z[NUM_TRAIN_DATA][NUM_OUTPUT_DIM];//教師データの出力
double test_z[NUM_TEST_DATA][NUM_OUTPUT_DIM];//テストデータの出力

double v[NUM_INPUT_DIM][NUM_OUTPUT_DIM];//出力層の重み
double delta_w[NUM_HIDDEN][NUM_INPUT_DIM][NUM_INPUT_DIM];//エッジの重み,w[層][from][to]
double delta_v[NUM_INPUT_DIM][NUM_OUTPUT_DIM];//出力層の重み

//評価用の変数
double train_err[NUM_TRAIN_DATA];//出力の評価関数
double train_overall_err = 0.0;
double test_err[NUM_TRAIN_DATA];//出力の評価関数
double test_overall_err = 0.0;

double eta = 0.1;
int num_epoch = 100000;

//その他
double tmp = 0.0;
int layer = 0;

int main(int argc, char const *argv[])
{ 
   

//rand()の初期化
    srand((unsigned)time(NULL));

//初期化
    for(int i=0; i<NUM_HIDDEN; i++) {
        for(int j=0; j<NUM_INPUT_DIM; j++) {
            for (int k = 0; k < NUM_INPUT_DIM; ++k){
                 w[i][j][k] = ((double)rand() / ((double)RAND_MAX + 1));
            }
            x[i][j] = 0.0;
            u[i][j] = 0.0;
        }
    }
    for (int j = 0; j < NUM_OUTPUT_DIM; ++j){
        for(int i = 0; i < NUM_INPUT_DIM; i++) {

            v[i][j] = ((double)rand() / ((double)RAND_MAX + 1));
        }
        for (int i = 0; i < NUM_TRAIN_DATA; ++i) {
            train_z[i][j] = 0.0;
        }
    }

//----------------------教師データで学習---------------------------
    for(int epoch = 0; epoch < num_epoch; epoch++) {


        //feedforward
        for(int data = 0; data < NUM_TRAIN_DATA; data++) {
            //隠れ層の計算
            for(int i=0; i<NUM_HIDDEN; i++) {
                for(int j=0; j<NUM_INPUT_DIM; j++) {
                    x[i][j] = 0.0;
                    u[i][j] = 0.0;
                }
            }
            for(layer = 0; layer < NUM_HIDDEN; layer++) {//layer番目の隠れ層の
                for (int to = 0; to < NUM_INPUT_DIM; to++){//m番目の次元の計算
                    if (layer == 0){//初めの層ならば、入力と内積を取る
                        for(int from = 0; from < NUM_INPUT_DIM; from++) {
                           tmp += train_x[data][from] * w[layer][from][to];
                        }
                    } else{
                        for(int from = 0; from < NUM_INPUT_DIM; from++) {
                           tmp += x[layer-1][from] * w[layer][from][to];
                        }
                    }
                    u[layer][to] = tmp;
                    x[layer][to] = sigmoid(tmp);//第l層のm番目の出力
                    tmp = 0;
                }
            }
            //出力層の計算
            for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
                for(int j = 0; j < NUM_INPUT_DIM; j++) {
                    tmp += x[layer][j] * v[j][i];
                }
                train_z[data][i] = tmp;//sigmoidしなくても良いっぽい？
                tmp = 0;
                //printf("to:%d %lf\n",i,train_z[data][i] );
            }




            //backward
            //出力層のdeltaを求める
            for(int to = 0; to < NUM_OUTPUT_DIM; to++) {
                for (int from = 0; from < NUM_INPUT_DIM; ++from){
                    delta_v[from][to] = train_z[data][to]*(train_z[data][to] -train_d[data][to]);   
                }
            }

            //隠れ層のdeltaを求める
            for (layer = NUM_HIDDEN-1; layer >= 0; layer--){
                if(layer == NUM_HIDDEN-1){
                    for(int to = 0; to < NUM_INPUT_DIM; to++) {
                        for (int from = 0; from < NUM_INPUT_DIM; ++from){
                            for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
                                tmp += delta_v[to][i] * v[to][i];
                            }
                            tmp *= d_sigmoid(u[layer][to]);
                            delta_w[layer][from][to] =tmp;
                            tmp = 0.0;  
                        }
                    }
                }else {
                    for(int to = 0; to < NUM_INPUT_DIM; to++) {
                        for (int from = 0; from < NUM_INPUT_DIM; ++from){
                            for (int i = 0; i < NUM_INPUT_DIM; ++i){
                                tmp += delta_w[layer + 1][to][i]*w[layer + 1][to][i];
                            }
                            tmp *= d_sigmoid(u[layer][to]); 
                            delta_w[layer][from][to] =tmp;
                            tmp = 0.0;
                        }
                    }
                }
            }
            //隠れ層の重みを更新
            for (layer = 0; layer < NUM_HIDDEN; ++layer){
                if (layer == 0){
                    for (int to = 0; to < NUM_INPUT_DIM; ++to){
                        for (int from = 0; from < NUM_INPUT_DIM; ++from){
                            w[layer][from][to] -= eta * train_x[data][from] * delta_w[layer][from][to]; 
                        }
                    }
                }else{
                    for (int to = 0; to < NUM_INPUT_DIM; ++to){
                        for (int from = 0; from < NUM_INPUT_DIM; ++from){
                            w[layer][from][to] -= eta * x[layer-1][from] * delta_w[layer][from][to]; 
                        }
                    }
                }
            }

            //出力層の重みを更新
            for (int to = 0; to < NUM_OUTPUT_DIM; ++to){
                for (int from = 0; from < NUM_INPUT_DIM; ++from){
                    v[from][to] -= eta * x[layer][from] * delta_v[from][to]; 
                }
            }
        }
        train_overall_err = 0.0;
        for (int data = 0; data < NUM_TRAIN_DATA; ++data){
            train_err[data] = 0.0;
            
            for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
                train_err[data] += pow((train_d[data][i]-train_z[data][i]),2);
            //printf("%lf:",train_z[data][i]);
            }
            train_err[data] /= 2.0;
            train_overall_err += train_err[data];
        }

        //print detail
        printf("epoch: %d,train_overall_err:%lf\n",epoch,train_overall_err);
        fprintf(stderr,"\repoch: %d,train_overall_err:%lf\n",epoch,train_overall_err);
    }
//----------------------テストデータで学習を評価--------------------------------
    for(int data = 0; data < NUM_TRAIN_DATA; data++) {
        //printf("data :%d\n",data);
        //-------------------隠れ層の計算--------------------------

        //-----------------ノードを初期化---------------------------------
        for(int i=0; i<NUM_HIDDEN; i++) {
            for(int j=0; j<NUM_INPUT_DIM; j++) {
                x[i][j] = 0.0;
                u[i][j] = 0.0;
            }
        }
        for(layer = 0; layer < NUM_HIDDEN; layer++) {//layer番目の隠れ層の
            for (int to = 0; to < NUM_INPUT_DIM; to++){//to番目の次元の計算
                if (layer == 0){//初めの層ならば、入力と内積を取る
                    for(int from = 0; from < NUM_INPUT_DIM; from++) {
                        tmp += test_x[data][from] * w[layer][from][to];
                    }
                } else{
                    for(int from = 0; from < NUM_INPUT_DIM; from++) {
                        tmp += x[layer-1][from] * w[layer][from][to];
                    }
                }
                u[layer][to] = tmp;
                x[layer][to] = sigmoid(tmp);//第l層のm番目の出力
                tmp = 0;
            }
        }
        //出力層の計算
        for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
            for(int j = 0; j < NUM_INPUT_DIM; j++) {
                tmp += x[layer][j] * v[j][i];
            }
            test_z[data][i] = tmp;//sigmoidしなくても良いっぽい？
            tmp = 0;
            //printf("to:%d %lf\n",i,test_z[data][i] );
        }

    }
    test_overall_err = 0.0;
    for (int data = 0; data < NUM_TRAIN_DATA; ++data){
        test_err[data] = 0.0;
        
        for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
            test_err[data] += pow((train_d[data][i]-test_z[data][i]),2);
        //printf("%lf:",test_z[data][i]);
        }
        test_err[data] /= 2.0;
        test_overall_err += train_err[data];
    }

    //print detail
    printf("train_overall_err:%lf\n",train_overall_err);

    return 0;
}
