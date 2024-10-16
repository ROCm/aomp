/* Scan error output from the compilation of "test_imperfect_loop.cxx".
   Every problem is a hammer here.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (void) {
  FILE *f = fopen ("test_imperfect_loop.ERR", "r");
  size_t bufsize = 1024;
  char *buf = (char*) malloc (bufsize);
  bool found = false;

  if (!f) {
    perror ("fopen");
    goto fail;
  }
  while (!feof (f) && !found) {
    getline (&buf, &bufsize, f);
    if (strstr (buf, "remark: Collapsing imperfectly-nested loop may "
                     "introduce unexpected data dependencies"))
      found = true;
  }
  fclose (f);

  if (found) {
    printf ("Imperfect loop warning test:: Pass\n");
    exit (0);
  }

fail:
  printf ("Imperfect loop warning test:: FAIL\n");
  return 1;
}
