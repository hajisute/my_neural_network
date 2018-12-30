#include "stdio.h"
#include "math.h"
#include "time.h"
#include "stdlib.h"
#include "jpeglib.h"
#include"float.h"
#include "string.h"
//num of units
#define NUM_INPUT_DIM 12*15+1 //120/10*150/10 + 1
#define NUM_OUTPUT_DIM 100
#define NUM_LAYER_DIM 12*15+1
#define NUM_LAYER_DEPTH 1
const int  NUM_TRAIN_DATA  = 496;
#define NUM_TEST_DATA 124
#define  K  5
#define EPSILON 0.000000000000000000000000000000000001

// double sigmoid(double x) {
//     return 1/(1+exp(-x));
// }

// //derivative of sigmoid function
// double d_sigmoid(double x) {
//     double a = 0.1;
//     return a*x*(1-x);
// } 
double ReLU(double x) {
    return (x >= 0) ? x :0.0 ;
}

//derivative of sigmoid function
double d_ReLU(double x) {
    return (x > 0) ? 1.0 : 0.0 ;
} 

int get_x(const char *filename, double train_x[NUM_INPUT_DIM]);
void get_d(char *filename,double train_d[NUM_OUTPUT_DIM],int data);
void feedforward(double input[NUM_INPUT_DIM], double output[NUM_OUTPUT_DIM], double w[NUM_LAYER_DEPTH][NUM_LAYER_DIM][NUM_LAYER_DIM], double x[NUM_LAYER_DEPTH][NUM_LAYER_DIM], double u[NUM_LAYER_DEPTH][NUM_LAYER_DIM]);
void backward(double input[NUM_INPUT_DIM], double output[NUM_OUTPUT_DIM], double d[NUM_OUTPUT_DIM],double w[NUM_LAYER_DEPTH][NUM_LAYER_DIM][NUM_LAYER_DIM], double v[NUM_LAYER_DIM][NUM_OUTPUT_DIM],
double x[NUM_LAYER_DEPTH][NUM_LAYER_DIM], double u[NUM_LAYER_DEPTH][NUM_LAYER_DIM]);



//train data
double train_x[NUM_INPUT_DIM];
double train_d[NUM_OUTPUT_DIM];
//test data
double test_x[NUM_INPUT_DIM];
double test_d[NUM_OUTPUT_DIM];

//ニューラルネット
double w[NUM_LAYER_DEPTH][NUM_LAYER_DIM][NUM_LAYER_DIM];//エッジの重み,w[層][from][to]

double x[NUM_LAYER_DEPTH][NUM_LAYER_DIM];//隠れ層の出力
double u[NUM_LAYER_DEPTH][NUM_LAYER_DIM];//隠れ層の入力
double train_z[NUM_OUTPUT_DIM];//教師データの出力
double test_z[NUM_OUTPUT_DIM];//テストデータの出力


double v[NUM_LAYER_DIM][NUM_OUTPUT_DIM];//出力層の重み

//評価用の変数
double train_err;//出力の評価関数
double train_overall_err = 0.0;
double test_err;//出力の評価関数
double test_overall_err[K] = {0};
double err = 0.0;

//精度の計算用
double train_acc;
double test_acc[K] = {0};
double acc = 0.0;

double max_d = DBL_MIN;
double max_z = DBL_MIN;
double argmax_d = DBL_MIN;
double argmax_z = DBL_MIN;

double eta = 0.01;
double alpha = 0.0000001;

int num_epoch = 1000000;

int main(int argc, char const *argv[])
{ 

    //rand()の初期化
    srand((unsigned)time(NULL));
    for(int sample = 0; sample < K; sample++){
        //----------------------教師データで学習---------------------------
        printf("Start Train:%d\n",sample);
        //初期化
        for(int i=0; i<NUM_LAYER_DEPTH; i++) {
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
                train_z[j] = 0.0;
            }
        }   
        train_overall_err = 0.0; 
        
        for(int epoch = 0; epoch < num_epoch; epoch++){
            train_acc = 0;
            //教師データで学習
            for(int data = 1; data <= NUM_TEST_DATA+NUM_TRAIN_DATA; data++) {
            
                max_d = DBL_MIN;
                max_z = DBL_MIN;
                argmax_d = DBL_MIN;
                argmax_z = DBL_MIN;

                if(data % K == sample)continue;
                char filename_img[100];
                sprintf(filename_img,"./img2/face%d.jpg",data);
                if( get_x(filename_img,train_x) )continue;
                get_d("./giin.csv",train_d,data);


                feedforward(train_x,train_z,w,x,u);

                backward(train_x,train_z,train_d,w,v,x,u);

                
                //損失を計算
                train_err = 0.0;
                for (int i = 0; i < NUM_OUTPUT_DIM; ++i)
                    train_err -= train_d[i] * log (EPSILON + train_z[i]);
                train_overall_err += train_err; 
                //printf("\rdata %d ,epoch:%d train_err:%lf",data,epoch,train_err);
                
                for(size_t i = 0; i < NUM_OUTPUT_DIM; i++){
                    //printf("epoch:%d, data:%d,d:%lf,z:%lf train_err:%lf\n",epoch,data,train_d[i],train_z[i],train_err);
                }

                //精度を計算
                for(size_t i = 0; i < NUM_OUTPUT_DIM; i++){
                    if(max_d <= train_d[i]) {
                        argmax_d = i;
                        max_d = train_d[i];
                    }
                    if(max_z <= train_z[i]) {
                        argmax_z = i;
                        max_z = train_z[i];
                    }
                }
                if (argmax_d == argmax_z) {
                    train_acc++;
                }
            }
            train_overall_err /= (double)NUM_TRAIN_DATA;

            //テストデータで評価

            test_acc[sample] = 0.0;
            test_overall_err[sample] = 0.0;

            for(int data = 1; data <= NUM_TEST_DATA+NUM_TRAIN_DATA; data++) {
                max_d = DBL_MIN;
                max_z = DBL_MIN;
                argmax_d = DBL_MIN;
                argmax_z = DBL_MIN;


                if(data % K != sample)continue;
                char filename_img[100];
                sprintf(filename_img,"./img2/face%d.jpg",data);
                get_x(filename_img,test_x);
                get_d("./giin.csv",test_d,data);



                feedforward(test_x,test_z,w,x,u);

                //損失を計算
                test_err = 0.0;
                for (int i = 0; i < NUM_OUTPUT_DIM; ++i)
                    test_err -= test_d[i] * log (EPSILON + test_z[i]);
                test_overall_err[sample] += test_err; 
                
                //精度を計算

                for(size_t i = 0; i < NUM_OUTPUT_DIM; i++){
                    if(max_d <= test_d[i]) {
                        argmax_d = i;
                        max_d = test_d[i];
                    }
                    if(max_z <= test_z[i]) {
                        argmax_z = i;
                        max_z = test_z[i];
                    }
                }
                if (argmax_d == argmax_z) {
                    test_acc[sample]++;
                }

            }
            //print detail
            printf("sample :%d, epoch %d ,test_overall_err:%lf, test_acc: %lf  ||  train_overall_err:%lf, train_acc: %lf\n",sample,epoch,test_overall_err[sample],test_acc[sample]/(double)NUM_TEST_DATA ,train_overall_err,train_acc/(double)NUM_TRAIN_DATA );
        }
            printf("sample :%d, train_overall_err:%lf, train_acc: %lf\n",sample,train_overall_err,train_acc/(double)NUM_TRAIN_DATA );        
        
    //----------------------テストデータで学習を評価--------------------------------
   
    }
    
    for(size_t i = 0; i < K; i++){
        err += test_overall_err[i];
        acc += test_acc[i];
    }
    err /= (double)K;
    acc /= (double)K;
    
    printf("cross valid : err %lf , acc %lf",err,acc);
    return 0;
}


int get_x(const char *filename, double train_x[NUM_INPUT_DIM])
{
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
    JSAMPARRAY img;

	FILE *fp;


	int i, j;
    int counter = 0;
	int width;
	int height;

	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_decompress( &cinfo );

	if( (fp = fopen( filename, "rb" ) )  == NULL ){
        fprintf(stderr,"\nerr occured.whlile opening :%s\n",filename);
        printf("aaa");
        return 1;
    }
	jpeg_stdio_src( &cinfo, fp );

	jpeg_read_header( &cinfo, TRUE );

	jpeg_start_decompress( &cinfo );

	width = cinfo.output_width;
	height = cinfo.output_height;


	img = (JSAMPARRAY)malloc( sizeof( JSAMPROW ) * height );
	for ( i = 0; i < height; i++ ) {
		img[i] = (JSAMPROW)calloc( sizeof( JSAMPLE ), 3 * width );
	}

	while( cinfo.output_scanline < cinfo.output_height ) {
		jpeg_read_scanlines( &cinfo,
			img + cinfo.output_scanline,
			cinfo.output_height - cinfo.output_scanline
		);
	}
    

	jpeg_finish_decompress( &cinfo );

	jpeg_destroy_decompress( &cinfo );

	fclose( fp );
	for ( i = 0; i < height; i = i + 10 ){
		for ( j = 0; j < width; j = j + 10 ) {
				train_x[counter] = (double)img[i][j]/255.0;
                counter++;
                //printf("%lf ", train_x[i * width + j]  );
		}
        //printf("\n");
	}
    train_x[NUM_INPUT_DIM - 1] = 1.0;
    for ( i = 0; i < height; i++ )
        free( img[i] );
	free( img );

    return 0;
}


void get_d(char *filename,double train_d[NUM_OUTPUT_DIM],int data)
{

    char buf[100] = {0};
    char *buf1;
    FILE *fp = fopen(filename, "r"); 
    for(size_t i = 0; i < data; i++){
        fgets(buf,100 - 1,fp);
    }
    fgets(buf,100 - 1,fp);
    strtok((unsigned char*)buf,",");
    strtok(NULL,",");
    strtok(NULL,",");
    buf1 = (char *)strtok(NULL,",");
    
    for(size_t i = 0; i < NUM_OUTPUT_DIM; i++){
        if(i == atoi(buf1))
            train_d[i]  = 1.0;
        else
            train_d[i] = 0.0;
    }
    
    

    fclose(fp);
}


void feedforward(double input[NUM_INPUT_DIM], double output[NUM_OUTPUT_DIM], double w[NUM_LAYER_DEPTH][NUM_LAYER_DIM][NUM_LAYER_DIM], double x[NUM_LAYER_DEPTH][NUM_LAYER_DIM], double u[NUM_LAYER_DEPTH][NUM_LAYER_DIM])
{

    double tmp = 0.0;
    double max = DBL_MIN;
    double sum = 0.0;

    
    //隠れ層の計算
    for(int i=0; i<NUM_LAYER_DEPTH; i++) {
        for(int j=0; j<NUM_INPUT_DIM; j++) {
            x[i][j] = 0.0;
            u[i][j] = 0.0;
        }
    }
    for(int layer = 0; layer < NUM_LAYER_DEPTH; layer++) {//layer番目の隠れ層の
        for (int to = 0; to < NUM_INPUT_DIM; to++){//m番目の次元の計算
            if(to != NUM_INPUT_DIM-1){
            
                if (layer == 0){//初めの層ならば、入力と内積を取る
                    for(int from = 0; from < NUM_INPUT_DIM; from++) {
                        tmp += input[from] * w[layer][from][to];
                    }
                } else{
                    for(int from = 0; from < NUM_INPUT_DIM; from++) {
                    tmp += x[layer-1][from] * w[layer][from][to];
                    }
                }
                 u[layer][to] = tmp;
                 x[layer][to] = ReLU(tmp);//第l層のm番目の出力
                 tmp = 0;
            }else{
                u[layer][to] = 0.0;
                x[layer][to] = 1.0;
            }
        }
    }

    for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
        for(int j = 0; j < NUM_INPUT_DIM; j++) {
            tmp += x[NUM_LAYER_DEPTH - 1][j] * v[j][i];
        }
        output[i] = tmp;//softmaxを計算
        if(max <= output[i]) max = output[i];
        tmp = 0;
    }
    
    for(size_t i = 0; i < NUM_OUTPUT_DIM; i++)
    {
        sum += exp(output[i] - max); 
        //printf("%lf\n",output[i] - max);
    }
    
    for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
        output[i] = exp(output[i] - max)/sum;
    }

    // //出力層の計算
    // for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
    //     for(int j = 0; j < NUM_INPUT_DIM; j++) {
    //         tmp += x[NUM_LAYER_DEPTH - 1][j] * v[j][i];
    //     }
    //     output[i] = tmp;//sigmoidしなくても良いっぽい？
    //     tmp = 0;
    // }


}

void backward(double input[NUM_INPUT_DIM], double output[NUM_OUTPUT_DIM], double d[NUM_OUTPUT_DIM],double w[NUM_LAYER_DEPTH][NUM_LAYER_DIM][NUM_LAYER_DIM], double v[NUM_LAYER_DIM][NUM_OUTPUT_DIM],
double x[NUM_LAYER_DEPTH][NUM_LAYER_DIM], double u[NUM_LAYER_DEPTH][NUM_LAYER_DIM])
{
    double tmp  = 0.0;
    double delta_w[NUM_LAYER_DEPTH][NUM_LAYER_DIM];//エッジの重み,w[層][from][to]
    double delta_v[NUM_OUTPUT_DIM];//出力層の重み
    double grad_w[NUM_LAYER_DEPTH][NUM_LAYER_DIM][NUM_LAYER_DIM];//エッジの重み,w[層][from][to]
    double grad_v[NUM_LAYER_DIM][NUM_OUTPUT_DIM];//出力層の重み

    //出力層のdeltaを求める
    for(int to = 0; to < NUM_OUTPUT_DIM; to++) {
        delta_v[to] = output[to] - d[to]; 
        //delta_v[from][to] = output[to]*(output[to] -d[to]);   
    }

    //隠れ層のdeltaを求める
    for (int layer = NUM_LAYER_DEPTH - 1; layer >= 0; layer--){
        if(layer == NUM_LAYER_DEPTH - 1){
            for(int to = 0; to < NUM_LAYER_DIM; to++) {
                for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
                    tmp += delta_v[i] * v[to][i];
                }
                tmp *= d_ReLU(u[layer][to]);
                delta_w[layer][to] = tmp;
                tmp = 0;
            }
        }else {
            for(int to = 0; to < NUM_LAYER_DIM; to++) {
                    for (int i = 0; i < NUM_LAYER_DIM - 1; ++i){
                        tmp += delta_w[layer + 1][i] * w[layer + 1][to][i];
                    }
                    tmp *= d_ReLU(u[layer][to]); 
                    delta_w[layer][to] = tmp;
                    tmp = 0.0;
                
            }
        }
    }
    
    //傾きを計算 & ノルムを計算
    double norm_grad = 0;
    for (int layer = 0; layer < NUM_LAYER_DEPTH; ++layer){
        if (layer == 0){
            for (int to = 0; to < NUM_LAYER_DIM - 1; ++to){
                for (int from = 0; from < NUM_LAYER_DIM; ++from){
                    if (from != NUM_LAYER_DIM - 1) {
                        grad_w[layer][from][to] =  input[from] * delta_w[layer][to];
                    }else {
                        grad_w[layer][from][to] =  delta_w[layer][to];
                    }
                    
                }   
            }
        }else{
            for (int to = 0; to < NUM_LAYER_DIM - 1; ++to){
                for (int from = 0; from < NUM_LAYER_DIM; ++from){
                    if (from != NUM_LAYER_DIM - 1) {
                        grad_w[layer][from][to] = x[layer - 1][from] * delta_w[layer][to];
                    }else {
                        grad_w[layer][from][to] =  delta_w[layer][to];
                    }
                }
            }
        }
    }
    for (int to = 0; to < NUM_OUTPUT_DIM; ++to){
        for (int from = 0; from < NUM_LAYER_DIM; ++from){
            if (from != NUM_LAYER_DIM - 1) {
                grad_v[from][to] = x[NUM_LAYER_DEPTH - 1][from] * delta_v[to];
            }else {
                grad_v[from][to] =  delta_v[to];
            }

        }
    }
    //隠れ層の重みを更新
    for (int layer = 0; layer < NUM_LAYER_DEPTH; ++layer){
        for (int to = 0; to < NUM_LAYER_DIM - 1; ++to){
            for (int from = 0; from < NUM_LAYER_DIM; ++from){
                w[layer][from][to] -= eta * grad_w[layer][from][to] + alpha * fabs(w[layer][from][to]); 
            }
        }
    }

    //出力層の重みを更新
    for (int to = 0; to < NUM_OUTPUT_DIM; ++to){
        for (int from = 0; from < NUM_LAYER_DIM; ++from){
            v[from][to] -= eta * grad_v[from][to] + alpha * fabs(v[from][to]); 
        }
    }

}
