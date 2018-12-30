#include "stdio.h"
#include "math.h"
#include "time.h"
#include "stdlib.h"
#include "jpeglib.h"
#include "string.h"
//num of units
#define NUM_INPUT_DIM 181 //120/10*150/10 + 1
#define NUM_OUTPUT_DIM 1
#define NUM_HIDDEN_DIM 20
#define NUM_HIDDEN 5
const int  NUM_TRAIN_DATA  = 496;
#define NUM_TEST_DATA 124
#define K 5

// double sigmoid(double x) {
//     return 1/(1+exp(-x));
// }

// //derivative of sigmoid function
// double d_sigmoid(double x) {
//     double a = 0.1;
//     return a*x*(1-x);
// } 
double sigmoid(double x) {
    return (x >= 0) ? x :0.0 ;
}

//derivative of sigmoid function
double d_sigmoid(double x) {
    return (x > 0) ? 1.0 : 0.0 ;
} 

int get_x(const char *filename, double train_x[NUM_INPUT_DIM]);
void get_d(char *filename,double train_d[NUM_OUTPUT_DIM],int data);
 


//train data
double train_x[NUM_INPUT_DIM];
double train_d[NUM_OUTPUT_DIM];
//test data
double test_x[NUM_INPUT_DIM];
double test_d[NUM_OUTPUT_DIM];

//ニューラルネット
double w[NUM_HIDDEN][NUM_INPUT_DIM][NUM_INPUT_DIM];//エッジの重み,w[層][from][to]

double x[NUM_HIDDEN][NUM_INPUT_DIM];//隠れ層の出力
double u[NUM_HIDDEN][NUM_INPUT_DIM];//隠れ層の入力
double train_z[NUM_OUTPUT_DIM];//教師データの出力
double test_z[NUM_OUTPUT_DIM];//テストデータの出力


double v[NUM_INPUT_DIM][NUM_OUTPUT_DIM];//出力層の重み
double delta_w[NUM_HIDDEN][NUM_INPUT_DIM][NUM_INPUT_DIM];//エッジの重み,w[層][from][to]
double delta_v[NUM_INPUT_DIM][NUM_OUTPUT_DIM];//出力層の重み

//評価用の変数
double train_err;//出力の評価関数
double train_overall_err = 0.0;
double test_err;//出力の評価関数
double test_overall_err = 0.0;

double eta = 0.00001;
int num_epoch = 10;

//その他
double tmp = 0.0;
int layer = 0;

int main(int argc, char const *argv[])
{ 

//rand()の初期化
    srand((unsigned)time(NULL));


    
    for(int sample = 1; sample < K; sample++){
        srand((unsigned)time(NULL));


        //----------------------教師データで学習---------------------------
        printf("Start Train:%d\n",sample);
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
                train_z[j] = 0.0;
            }
        }

  


   
        train_overall_err = 0.0; 
        
        for(int epoch = 0; epoch < num_epoch; epoch++){
            /* code */
        
           
            for(int data = 1; data <= NUM_TEST_DATA+NUM_TRAIN_DATA; data++) {
                //feedforward

                if(data % K == sample)continue;
                char filename_img[100];
                sprintf(filename_img,"./img2/face%d.jpg",data);
                if( get_x(filename_img,train_x) )continue;
                
                //for(size_t i = 0; i < NUM_INPUT_DIM; i++){
                //    printf(" %lf ",train_x[i]);
                //}
                //printf("\n");
                get_d("./giin.csv",train_d,data);


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
                            tmp += train_x[from] * w[layer][from][to];
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
                        tmp += x[NUM_HIDDEN - 1][j] * v[j][i];
                    }
                    train_z[i] = tmp;//sigmoidしなくても良いっぽい？
                    tmp = 0;
                }




                //backward
                //出力層のdeltaを求める
                for(int to = 0; to < NUM_OUTPUT_DIM; to++) {
                    for (int from = 0; from < NUM_INPUT_DIM; ++from){
                        delta_v[from][to] = train_z[to]*(train_z[to] -train_d[to]);   
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
                                w[layer][from][to] -= eta * train_x[from] * delta_w[layer][from][to]; 
                            }
                        }
                    }else{
                        for (int to = 0; to < NUM_INPUT_DIM; ++to){
                            for (int from = 0; from < NUM_INPUT_DIM; ++from){
                                w[layer][from][to] -= eta * x[layer - 1][from] * delta_w[layer][from][to]; 
                            }
                        }
                    }
                }

                //出力層の重みを更新
                for (int to = 0; to < NUM_OUTPUT_DIM; ++to){
                    for (int from = 0; from < NUM_INPUT_DIM; ++from){
                        v[from][to] -= eta * x[NUM_HIDDEN - 1][from] * delta_v[from][to]; 
                    }
                }
                train_err = 0.0;
                //損失を計算
                for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
                    train_err += pow((train_d[i]-train_z[i]),2);
                
                fflush(stdout);
                train_err /= 2.0;
                train_overall_err += train_err; 
                //printf("\rdata %d ,epoch:%d train_err:%lf",data,epoch,train_err);
                printf("\repoch:%d, data:%d,d:%lf,z:%lf train_err:%lf",epoch,data,train_d[0],train_z[0],train_err);

                //精度を計算
                }

            }
            train_overall_err /= (double)NUM_TRAIN_DATA;
            printf("\nepoch %d ,train_overall_err:%lf\n",epoch,train_overall_err);
        
        }
        printf("sample %d,train_overall_err:%lf\n",sample,train_overall_err);
        
    //----------------------テストデータで学習を評価--------------------------------
        printf("Start Test:%d\n",sample);
        for(int data = 1; data <= NUM_TEST_DATA+NUM_TRAIN_DATA; data++) {
            if(data % K != sample)continue;
            char filename_img[100];
            sprintf(filename_img,"./img2/face%d.jpg",data);
            get_x(filename_img,test_x);
            get_d("./giin.csv",test_d,data);

            test_overall_err = 0.0;

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
                            tmp += test_x[from] * w[layer][from][to];
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
                    tmp += x[NUM_HIDDEN - 1][j] * v[j][i];
                }
                test_z[i] = tmp;//sigmoidしなくても良いっぽい？
                tmp = 0;
            }
            test_err = 0.0;
            for (int i = 0; i < NUM_OUTPUT_DIM; ++i){
                test_err += pow((train_d[i]-test_z[i]),2);
            }
            test_err /= 2.0;
            test_overall_err += train_err;
        }


        //print detail
        printf("sample:%d train_overall_err:%lf\n",sample,train_overall_err);
    }
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
    train_x[NUM_INPUT_DIM - 1] = 0.0;
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
    train_d[0]  = atoi(buf1);
    fclose(fp);
}


/*
double ** matmalloc(int row,int col)
{
    int i,j;
    double **ptr = (double **)malloc(sizeof(double *) * row);
    if (ptr == NULL) {
        fprintf(stderr,"error!\n");
        exit(1);
    }
    for (i=0;i<row;i++) {
        ptr[i] = (double *)malloc(sizeof(double)*col);
        if (ptr[i]==NULL) {
            fprintf(stderr,"error!\n");
            exit(1);
        }
        for (j=0;j<col;j++) ptr[i][j]=0.0;
    }

    return ptr;
}*/

