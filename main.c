/*********************************************
   gcc -O1 -fopenmp main.c -lrt -o main
   OMP_NUM_THREADS=4 ./main
*/

#include <stdio.h>
#include <stdlib.h>
#include "pthread.h"
#include <time.h>
#include <math.h>

#define A  8  /* coefficient of x^2 */
#define B  8  /* coefficient of x */
#define C  8  /* constant term */

#define NUM_TESTS 10
#define OPTIONS 3

typedef float data_t;
/* Create abstract data type for matrix */
typedef struct {
    long int rowlen;
    data_t *data;
} matrix_rec, *matrix_ptr;

typedef struct{
    int *x;
    int *y;
    data_t *data;
}sparse_matrix_rec, *sparse_matrix_ptr;

struct thread_data{
    int thread_id;
    matrix_ptr b;
    matrix_ptr c;
    int * row;
    int * col;
    data_t * sparse_a;
};

/* Prototypes */
double wakeup_delay();
double interval(struct timespec start, struct timespec end);
int clock_gettime(clockid_t clk_id, struct timespec *tp);
matrix_ptr new_matrix(long int rowlen);
sparse_matrix_ptr new_sparse_matrix(long int elements);
int set_matrix_rowlen(matrix_ptr m, long int rowlen);
long int get_matrix_rowlen(matrix_ptr m);
data_t *get_matrix_start(matrix_ptr m);
data_t *get_matrix_start_sp(sparse_matrix_ptr m);
int *get_matrix_start_x(sparse_matrix_ptr m);
int *get_matrix_start_y(sparse_matrix_ptr m);
int init_sparse_matrix(matrix_ptr m, long int rowlen, int n);
int get_sparse_matrix(matrix_ptr m, sparse_matrix_ptr s, long int rowlen);
int zero_matrix(matrix_ptr m, long int rowlen);
void print_matrix(matrix_ptr a);
long int count_nz(matrix_ptr m, long int rowlen);
void smm_serial2(sparse_matrix_ptr a, long int nz_a, matrix_ptr b, matrix_ptr c);
void smm_serial(sparse_matrix_ptr sp_a, sparse_matrix_ptr sp_b, matrix_ptr c, long int nz_a, long int nz_b);
void smm_serial3(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void smm_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void smm_parallel(matrix_ptr a, matrix_ptr b, matrix_ptr c);

#define THREADS 4
int NUM_THREADS = 2; // make this 4, 8 for running it in 4 core and 8 core machines

void detect_threads_setting()
{
    long int i, ognt;
    char * env_ONT;

    // Find out how many threads OpenMP thinks it is wants to use
#pragma omp parallel for
    for(i=0; i<1; i++) {
        ognt = omp_get_num_threads();
    }

    printf("omp's default number of threads is %d\n", ognt);

    // If this is illegal (0 or less), default to the "#define THREADS"
    //   value that is defined above
    if (ognt <= 0) {
        if (THREADS != ognt) {
            printf("Overriding with #define THREADS value %d\n", THREADS);
            ognt = THREADS;
        }
    }
    omp_set_num_threads(ognt);
    // Once again ask OpenMP how many threads it is going to use
#pragma omp parallel for
    for(i=0; i<1; i++) {
        ognt = omp_get_num_threads();
    }
    printf("Using %d threads for OpenMP\n", ognt);
}


int main() {
    int debug = 0; //Prints the matrices & resulting matrix for user to validate

    int OPTION;
    struct timespec time_start, time_stop;
    double time_stamp[OPTIONS][NUM_TESTS];
    double final_answer;
    long int x, n, i, j, alloc_size;

    printf("Sparse Matrix Multiply\n");
    final_answer = wakeup_delay();
    //detect_threads_setting();
    /* declare and initialize the matrix structures */
    x = NUM_TESTS-1;
    alloc_size = A*x*x + B*x + C;

    // FIRST STEP: INIT TWO SPARSE MATRICES
    matrix_ptr a0 = new_matrix(alloc_size);
    init_sparse_matrix(a0, alloc_size,4); //most of the matrix has 0 elements
    matrix_ptr b0 = new_matrix(alloc_size);
    init_sparse_matrix(b0, alloc_size,4); //most of the matrix has 0 elements
    matrix_ptr c0 = new_matrix(alloc_size);
    zero_matrix(c0, alloc_size); //result matrix

    // LASTLY RUN THE TESTS
    for (OPTION = 0; OPTION<OPTIONS; OPTION++) {
        printf("Doing OPTION=%d...\n", OPTION);
        for (x = 0; x < NUM_TESTS && (n = A * x * x + B * x + C, n <= alloc_size); x++) {
            set_matrix_rowlen(a0, n);
            set_matrix_rowlen(b0, n);
            set_matrix_rowlen(c0,n);

            //long int nz_a = count_nz(a0,n);
            //long int nz_b = count_nz(b0,n);

            //sparse_matrix_ptr sp_a = new_sparse_matrix(nz_a);
            //sparse_matrix_ptr sp_b = new_sparse_matrix(nz_b);

            //get_sparse_matrix(a0, sp_a,n);
            //get_sparse_matrix(b0,sp_b,n);

            if(debug) {
                // PRINT THE MATRICES FOR GOOD VISUALIZATION & VALIDATION
                print_matrix(a0);
                print_matrix(b0);
            }
            clock_gettime(CLOCK_REALTIME, &time_start);
            switch(OPTION) {
                case 0:
                    smm_serial3(a0,b0,c0);
                    if(debug){printf("Result matrix:\n"); print_matrix(c0);}
                    break;
                case 1:
                    smm_parallel(a0,b0,c0);
                    if(debug){printf("Result matrix:\n"); print_matrix(c0);}
                    break;
                case 2:
                    NUM_THREADS = 4;
                    smm_parallel(a0,b0,c0);
                    if(debug){printf("Result matrix:\n"); print_matrix(c0);}
                    break;
                case 3:
                    NUM_THREADS = 8;
                    smm_parallel(a0,b0,c0);
                    if(debug){printf("Result matrix:\n"); print_matrix(c0);}
                    break;
                case 4:
                    smm_omp(a0,b0,c0);
                    if(debug){printf("Result matrix:\n"); print_matrix(c0);}
                    break;
                default: break;
            }
            clock_gettime(CLOCK_REALTIME, &time_stop);
            time_stamp[OPTION][x] = interval(time_start, time_stop);
            printf("  iter %d done\r", x); fflush(stdout);
        }
        printf("\n");
    }
    printf("\nAll times are in seconds\n");
    printf("rowlen, smm_serial, smm_parallel, smm_omp \n");
    {
        int i, j;
        for (i = 0; i < x; i++) {
            printf("%4ld",  A*i*i + B*i + C);
            for (j = 0; j < OPTIONS; j++) {
                printf(",%10.4g", time_stamp[j][i]);
            }
            printf("\n");
        }
    }
    printf("\n");
    printf("Initial delay was calculating: %g \n", final_answer);
} /* end main */

void print_matrix(matrix_ptr a)
{
    long int i, j;
    long int row_length = get_matrix_rowlen(a);
    data_t *a0 = get_matrix_start(a);
    for (i=0; i<row_length; i++) {
        printf("| ");
        for (j = 0; j < row_length-1; j++) {
            data_t val = a0[i*row_length+j];
            printf("%g, ",val) ;
        }
        data_t val = a0[i*row_length+j];
        printf("%g ", val);
        printf("|\n");
    }
    printf("\n");
}

/* Sparse matrix transformation & multiplication
 * Author: Sevval Simsek
 * */
void smm_serial(sparse_matrix_ptr sp_a, sparse_matrix_ptr sp_b, matrix_ptr c, long int nz_a, long int nz_b){
    long int i, j, k;
    long int row_length = get_matrix_rowlen(c);
    data_t *c0 = get_matrix_start(c);
    data_t *sp_a0 = get_matrix_start_sp(sp_a); // data: nonzero elements
    data_t *sp_b0 = get_matrix_start_sp(sp_b);

    int *x_a0 = get_matrix_start_x(sp_a); // x coordinates: row values
    int *x_b0 = get_matrix_start_x(sp_b);
    int *y_a0 = get_matrix_start_y(sp_a); // y coordinates: column values
    int *y_b0 = get_matrix_start_y(sp_b);

    for(i=0;i<nz_a;i++){ //iterate for all nonzero elements in A
        for(j=0;j<nz_b;j++){ //iterate for all nonzero elements in B
            if(y_a0[i]==x_b0[j])
            {
                int row = x_a0[i];
                int col = y_b0[j];
                c0[row*row_length+col] += sp_a0[i]*sp_b0[j];
            }
        }
    }
}

void smm_serial2(sparse_matrix_ptr a, long int nz_a, matrix_ptr b, matrix_ptr c){
    long int i,j,res;
    long int row_length = get_matrix_rowlen(b);

    int *x = get_matrix_start_x(a);
    int *y = get_matrix_start_y(a);

    data_t *a0 = get_matrix_start_sp(a);
    data_t *b0 = get_matrix_start(b);
    data_t *c0 = get_matrix_start(c);
    // a is nonzero element matrix, x,y are for the coordinates of nonzero elements
    // b, c regular nxn matrices
    for (i=0; i<nz_a;i++) {
        //get the element's coordinates
        int row = x[i];
        int col = y[i];
        for (j = 0; j < row_length; j++) {
            int index = col*row_length + j;
            if(b0[index]!=0) {
                res = a0[i] * b0[index];
                c0[row*row_length+j] += res;
            }
        }
    }
}

void smm_serial3(matrix_ptr a, matrix_ptr b, matrix_ptr c){
    long int i, j, k;
    long int row_length = get_matrix_rowlen(a);
    data_t *a0 = get_matrix_start(a);
    data_t *b0 = get_matrix_start(b);
    data_t *c0 = get_matrix_start(c);
    data_t sum;

    data_t sparse_a[(row_length*row_length)/4];
    int row[(row_length*row_length)/4];
    int col[(row_length*row_length)/4];
    k=0;
    // Getting the nonzero elements of A
     for(i=0; i<row_length; i++){
         for(j=0; j<row_length;j++){
             if(a0[i*row_length+j] != 0){
                 sparse_a[k]=a0[i*row_length+j];
                 row[k] = i;
                 col[k] = j;
                 k++;
             }
         }
     }
     // Multiply the nonzero elements in A with the corresponding elements in B
     for(i=0; i<row_length/4; i++){
         int row_A = row[i];
         int col_A = col[i];
         for(j=0;j<row_length; j++){
             if(b0[col_A * row_length+j] != 0)
                 c0[row_A * row_length +j] += sparse_a[i]*b0[col_A*row_length+j];
         }
     }
}

void smm_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c){
    long int i, j, k;
    long int row_length = get_matrix_rowlen(a);
    data_t *a0 = get_matrix_start(a);
    data_t *b0 = get_matrix_start(b);
    data_t *c0 = get_matrix_start(c);
    data_t sum;

    // Getting the nonzero elements of A

    data_t sparse_a[(row_length*row_length)/4];
    int row[(row_length*row_length)/4];
    int col[(row_length*row_length)/4];
    k=0;
    for (i = 0; i < row_length; i++) {
        for (j = 0; j < row_length; j++) {
            if (a0[i * row_length + j] != 0) {
                sparse_a[k] = a0[i * row_length + j];
                row[k] = i;
                col[k] = j;
                k++;
            }
        }
    }

    // Multiply the nonzero elements in A with the corresponding elements in B
#pragma omp parallel shared(sparse_a,b0,c0,row_length) private(i,j)
    {
        #pragma omp for
        for (i = 0; i < row_length / 4; i++) {
            int row_A = row[i];
            int col_A = col[i];
            for (j = 0; j < row_length; j++) {
                if (b0[col_A * row_length + j] != 0)
                    c0[row_A * row_length + j] += sparse_a[i] * b0[col_A * row_length + j];
            }
        }
    }
}

/*
 * How I parallelize this code:
 * each thread gets the whole array sparse_A and coordinates
 * multiplies that with n/NUM_THREADS columns from B
 * gets n/NUM_THREADS columns of C
 * then we reconstruct C (in runtime)
 * */
void *smm_work(void *threadarg){
    long int i, j, k, low, high, sum;
    struct thread_data *my_data;
    my_data = (struct thread_data *) threadarg;
    int taskid = my_data->thread_id;
    matrix_ptr b0 = my_data->b;
    matrix_ptr c0 = my_data->c;
    int *row = my_data->row;
    int *col = my_data->col;
    data_t *sparse_a = my_data->sparse_a;

    long int rowlen = get_matrix_rowlen(b0);
    data_t *bM = get_matrix_start(b0);
    data_t *cM = get_matrix_start(c0);

    low = (taskid * rowlen)/NUM_THREADS;
    high = ((taskid+1) * rowlen)/NUM_THREADS;
    //todo: actual work (eg. work on certain rows)

    for (i = 0; i < rowlen*rowlen / 4; i++) {
        int row_A = row[i];
        int col_A = col[i];
        for (j = low; j < high; j++) {
            if (bM[col_A * rowlen + j] != 0)
                cM[row_A * rowlen + j] += sparse_a[i] * bM[col_A * rowlen + j];
        }
    }
    pthread_exit(NULL);
}

void smm_parallel(matrix_ptr a, matrix_ptr b, matrix_ptr c){
    long int i, j, k;
    pthread_t threads[NUM_THREADS];
    struct thread_data thread_data_array[NUM_THREADS];
    int rc;
    long t;
    /* I have to extract the nonzero elements of matrix A before
     * passing the variable to the threads, so this operation
     * is excluded from parallelization.
     */
    long int row_length = get_matrix_rowlen(a);
    data_t *a0 = get_matrix_start(a);

    data_t sparse_a[(row_length*row_length)/4];
    int row[(row_length*row_length)/4];
    int col[(row_length*row_length)/4];
    k=0;

    // Getting the nonzero elements of A
    for(i=0; i<row_length; i++){
        for(j=0; j<row_length;j++){
            if(a0[i*row_length+j] != 0){
                sparse_a[k]=a0[i*row_length+j];
                row[k] = i;
                col[k] = j;
                k++;
            }
        }
    }

    for (t = 0; t < NUM_THREADS; t++) {
        thread_data_array[t].thread_id = t;
        thread_data_array[t].sparse_a = sparse_a;
        thread_data_array[t].row = row;
        thread_data_array[t].col = col;
        thread_data_array[t].b = b;
        thread_data_array[t].c = c;
        rc = pthread_create(&threads[t], NULL, smm_work,
                            (void*) &thread_data_array[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (t = 0; t < NUM_THREADS; t++) {
        if (pthread_join(threads[t],NULL)){
            printf("ERROR; code on return from join is %d\n", rc);
            exit(-1);
        }
    }
}

/****************** HELPER FUNCTIONS START **********************/
double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0) {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
    double meas = 0; int i, j;
    struct timespec time_start, time_stop;
    double quasi_random = 0;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    j = 100;
    while (meas < 1.0) {
        for (i=1; i<j; i++) {
            /* This iterative calculation uses a chaotic map function, specifically
               the complex quadratic map (as in Julia and Mandelbrot sets), which is
               unpredictable enough to prevent compiler optimisation. */
            quasi_random = quasi_random*quasi_random - 1.923432;
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        meas = interval(time_start, time_stop);
        j *= 2; /* Twice as much delay next time, until we've taken 1 second */
    }
    return quasi_random;
}

/* Create matrix of specified length */
matrix_ptr new_matrix(long int rowlen)
{
    long int i;

    /* Allocate and declare header structure */
    matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
    if (!result) return NULL;  /* Couldn't allocate storage */
    result->rowlen = rowlen;

    /* Allocate and declare array */
    if (rowlen > 0) {
        data_t *data = (data_t *) calloc(rowlen*rowlen, sizeof(data_t));
        if (!data) {
            free((void *) result);
            printf("COULD NOT ALLOCATE %ld BYTES STORAGE \n",
                   rowlen * rowlen * sizeof(data_t));
            exit(-1);
        }
        result->data = data;
    }
    else result->data = NULL;

    return result;
}

sparse_matrix_ptr new_sparse_matrix(long int elements){
    long int i;

    /* Allocate and declare header structure */
    sparse_matrix_ptr result = (sparse_matrix_ptr) malloc(sizeof(sparse_matrix_rec));
    if (!result) return NULL;  /* Couldn't allocate storage */

    /* Allocate and declare array */
    if (elements > 0) {
        data_t *data = (data_t *) calloc(elements, sizeof(data_t));
        int *x = (int *) calloc(elements, sizeof(int));
        int *y = (int *) calloc(elements, sizeof(int));
        if (!data || !x || !y) {
            free((void *) result);
            printf("COULD NOT ALLOCATE %ld BYTES STORAGE \n",
                   elements * sizeof(data_t));
            exit(-1);
        }
        result->data = data;
        result->x = x;
        result->y = y;
    }
    else {
        result->data = NULL;
        result->x = NULL;
        result->y = NULL;
    }

    return result;
}

/* Set row length of matrix */
int set_matrix_rowlen(matrix_ptr m, long int rowlen)
{
    m->rowlen = rowlen;
    return 1;
}

/* Return row length of matrix */
long int get_matrix_rowlen(matrix_ptr m)
{
    return m->rowlen;
}

/* initialize sparse matrix where every 1 element in n elements is nonzero
int init_sparse_matrix(matrix_ptr m, long int rowlen, int n)
{
    long int i;
    if (rowlen > 0) {
        m->rowlen = rowlen;
        for (i = 0; i < rowlen*rowlen; i++)
            if(i%n == 0)
                m->data[i] = (data_t)(i);
            else
                m->data[i] = 0;
        return 1;
    }
    else return 0;
}
*/

/* initialize sparse matrix where every 1 element in n elements is nonzero */
int init_sparse_matrix(matrix_ptr m, long int rowlen, int n)
{
    long int i,j;
    if (rowlen > 0) {
        m->rowlen = rowlen;
        for (i = 0; i < rowlen; i++){
            for (j=0; j<rowlen; j++){
                if ( i == j)
                    m->data[i*rowlen+j] = (data_t) (i);
                else if ((i+j)%n==0)
                    m->data[i] = (data_t) (i);
                else
                    m->data[i] = 0;
            }
        }
        return 1;
    }
    else return 0;
}


/* initialize matrix */
int zero_matrix(matrix_ptr m, long int rowlen)
{
    long int i,j;

    if (rowlen > 0) {
        m->rowlen = rowlen;
        for (i = 0; i < rowlen*rowlen; i++) {
            m->data[i] = 0;
        }
        return 1;
    }
    else return 0;
}

/*
 * Creates a vector of sparse matrix elements.
 * The coordinates (x,y) values are stored in the same index as the data values.
 */
int get_sparse_matrix(matrix_ptr m, sparse_matrix_ptr s, long int rowlen){
    long int i, j=0;

    if(rowlen>0){
        for (i = 0; i < rowlen*rowlen; i++) {
            if (m->data != 0) {
                s->x[j] = (int) i/rowlen;
                s->y[j] = (int) i%rowlen;
                s->data[j] = m->data[i];
                j++;
            }
        }
        return j;  //return the number of nonzero elements
    }
    return 0;
}

data_t *get_matrix_start(matrix_ptr m)
{
    return m->data;
}
data_t *get_matrix_start_sp(sparse_matrix_ptr m)
{
    return m->data;
}
int *get_matrix_start_x(sparse_matrix_ptr m){
    return m->x;
}
int *get_matrix_start_y(sparse_matrix_ptr m){
    return m->y;
}

long int count_nz(matrix_ptr m, long int rowlen)
{
    long int i,count=0;
    data_t *m0 = get_matrix_start(m);
    for(i=0;i<rowlen*rowlen;i++)
    {
        if(m0[i]!=0)
            count++;
    }
    return count;
}
/********************** HELPER FUNCTIONS END***************************/




