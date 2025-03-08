// Input is assumed to be valid

typedef struct {
  int rows, cols; // # of rows and columns
  float data;     // row by row flattened and extends past the struct
} matrix;

#define elem(a, b, c) ((&(a->data))[(a->cols * b) + c])
// matrix will be 0-indexed for the sake of simplicity

kernel void mlt(global matrix *a, global matrix *b, global matrix *c/*,
                global float *debug*/) {
  printf("Run mlt!!!\n");
  // printf("debug[0] = %f\n", debug[0]);
  // printf("debug=%d, a->data=%d\n", (int)debug, a->data);
  printf("a->rows=%d,a->cols=%d\n", a->rows, a->cols);
  // printf("a->data=%p\n", a->data);
  int ind = get_global_id(0);
  int row = ind / c->cols;
  int col = ind % c->cols;
  (&(c->data))[ind] = 0;
  printf("Da init value for %d: %f\n", ind, (&(c->data))[ind]);
  printf("a's data's first number is %0X\n",
         *((global uint *)((global void *)(&((&(a->data))[0])))));
  for (int i = 0; i < a->rows; i++) {
    printf("its %f * %f\n", elem(a, row, i), elem(b, i, col));
    (&(c->data))[ind] += elem(a, row, i) * elem(b, i, col);
  }
  printf("Da value for %d: %f\n", ind, (&(c->data))[ind]);
  printf("DONE WOOOOOOOOOOO!\n");
}

kernel void add(global matrix *a, global matrix *b, global matrix *c) {
  int ind = get_global_id(0);
  (&(c->data))[ind] = (&(a->data))[ind] + (&(b->data))[ind];
}

kernel void sub(global matrix *a, global matrix *b, global matrix *c) {
  int ind = get_global_id(0);
  (&(c->data))[ind] = (&(a->data))[ind] - (&(b->data))[ind];
}

kernel void trans(global matrix *a, global matrix *b) {
  int ind = get_global_id(0);
  int row = ind / a->cols;
  int col = ind % a->cols;
  elem(b, col, row) = elem(a, row, col);
}

kernel void relu(global matrix *a, global matrix *b) {
  // Leaky ReLU
  int ind = get_global_id(0);
  float itm = (&(a->data))[ind];
  (&(b->data))[ind] = itm >= 0 ? itm : 0.01 * itm;
}

kernel void drelu(global matrix *a, global matrix *b) {
  // Leaky ReLU's derivitive
  int ind = get_global_id(0);
  float itm = (&(a->data))[ind];
  (&(b->data))[ind] = itm >= 0 ? 1 : 0.01;
}

kernel void clip(global matrix *a) {
  // Gradient clipper
  int ind = get_global_id(0);
  float itm = (&(a->data))[ind];
  (&(a->data))[ind] = itm >= 0 ? fmin(1.0f,itm): fmax(-1.0f,itm);
}

uint randomInt(uint ind){
  // Uses xorshift and an LCG
  // LCG
  uint tmp=ind*1664525u+1013904223u;
  // xorshift
  tmp^=tmp<<13;
  tmp^=tmp>>17;
  tmp^=tmp<<5;
  printf("random: %d\n",tmp);
  return tmp;
}

kernel void heInit(global matrix* a, int nrs){
  // He initialization
  // nrs is number of neurons
  uint ind=get_global_id(0);
  float u1=((float)(randomInt(ind)>>8))/((float)0x1000000);
  float u2=((float)(randomInt(ind*1664525u+1013904223u)>>8))/((float)0x1000000);
  printf("u1: %f",u1);
  printf("u2: %f",u2);
  (&(a->data))[ind]=sqrt(-2.0f*log(u1))*cospi(2.0f*u2)*sqrt(2.0f/((float)nrs));
}

kernel void xInit(global matrix* a, int nrs){
  // Xavier initialization
  // nrs is number of neurons
  uint ind=get_global_id(0);
  float u1=((float)(randomInt(ind)>>8))/((float)0x1000000);
  float u2=((float)(randomInt(ind*1664525u+1013904223u)>>8))/((float)0x1000000);
  printf("u1: %f",u1);
  printf("u2: %f",u2);
  (&(a->data))[ind]=sqrt(-2.0f*log(u1))*cospi(2.0f*u2)*sqrt(1.0f/((float)nrs));
}

kernel void sig(global matrix *a, global matrix *b) {
  // Sigmoid
  int ind = get_global_id(0);
  float itm = (&(a->data))[ind];
  (&(b->data))[ind] = 1.0f / (1.0f + exp(-itm));
}
