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